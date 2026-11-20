"""Planning and validation for coalesced multi-range reads.

This module contains the I/O-free logic behind the ``range_policy``
option of :meth:`fsspec.implementations.http.HTTPFileSystem.cat_ranges`
and of :class:`fsspec.implementations.reference.ReferenceFileSystem`.

A *range policy* is a dict with exactly the keys

- ``max_ranges``: maximum number of byte intervals declared in a single
  HTTP ``Range`` request (non-bool int, >= 1),
- ``max_request_bytes``: maximum total number of source bytes declared
  in a single request (non-bool int, >= 1),
- ``extra_bytes``: call-level budget for bytes fetched beyond the union
  of the requested ranges, used to bridge gaps between neighbouring
  ranges (non-bool int, >= 0).

Given a batch of ``(url, start, end)`` ranges (end exclusive), the
planner groups them by URL, computes the union of the needed bytes per
URL (of total length ``U``) and emits a list of physical requests such
that

- every request declares at most ``max_ranges`` disjoint intervals,
- the intervals of a request sum to at most ``max_request_bytes``,
- the intervals of *all* requests sum to at most ``U + extra_bytes``.

Ranges longer than ``max_request_bytes`` are split into several
requests. The planner never downloads the same byte twice; duplicate or
contained input ranges are fetched once and mapped back to each input.

A single :class:`CoalesceBudget` holds the ``extra_bytes`` allowance of
one whole call. A batch whose URLs are served by independent backends is
planned in per-backend pieces against that one shared object, and the
same object is later used to validate server-coalesced responses: the
planner's gap bridging and every accepted response draw from one pool,
so no backend or execution batch can claim the allowance twice.

The planning strategy is a per-URL left-to-right greedy sweep: union
intervals are accumulated into a segment until either the byte budget
of the segment is exhausted (the segment is closed) or the interval
count exceeds ``max_ranges``, in which case the smallest gaps inside
the segment are *bridged* (paid for from the shared ``extra_bytes``
budget) to merge intervals back under the limit. See
``docs/source/features.rst`` for a comparison with the considered
alternative (global cheapest-gap pre-bridging plus first-fit packing).
"""

import bisect
import heapq
import numbers
from typing import NamedTuple


class RangePolicyError(ValueError):
    """Invalid ``range_policy`` specification or invalid range inputs.

    Raised before any request is issued.
    """


class RangeResponseError(IOError):
    """A multi-range response was invalid, inconsistent or over budget."""


class CoalesceBudget:
    """The single ``extra_bytes`` budget of one whole batch.

    One instance is shared by every physical request of a call, even when
    the references resolve to different backend filesystems. The planner
    draws on it first (the gap-bridging bytes of the declared ranges), and
    each accepted response draws on it again for the source bytes a
    server coalesced beyond what that request declared. Check-and-take
    contains no awaits, so concurrent requests cannot oversubscribe it.

    ``remaining`` is the budget still available for either purpose; the
    total of planner-drawn and response-drawn bytes can therefore never
    exceed the original ``extra_bytes``.
    """

    __slots__ = ("remaining",)

    def __init__(self, remaining):
        self.remaining = remaining

    def take(self, n, what="the extra_bytes budget"):
        """Charge ``n`` source bytes, or raise before changing the state."""
        if n > self.remaining:
            raise RangeResponseError(f"coalesced response exceeds {what}")
        self.remaining -= n


_POLICY_KEYS = ("max_ranges", "max_request_bytes", "extra_bytes")


def _check_positive_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RangePolicyError(
            f"range_policy {name!r} must be an integer, got {value!r}"
        )
    value = int(value)
    if value < minimum:
        raise RangePolicyError(
            f"range_policy {name!r} must be >= {minimum}, got {value}"
        )
    return value


def validate_range_policy(policy):
    """Validate a range policy and return ``(max_ranges, max_request_bytes,
    extra_bytes)``.

    Raises
    ------
    RangePolicyError (a ValueError)
        If the policy is not a dict, has missing or unknown keys, or
        any value is not a non-bool integer within its allowed range.
    """
    if not isinstance(policy, dict):
        raise RangePolicyError(
            f"range_policy must be a dict with keys {_POLICY_KEYS}, got {policy!r}"
        )
    unknown = sorted(set(policy).difference(_POLICY_KEYS))
    if unknown:
        raise RangePolicyError(f"unknown range_policy keys: {unknown}")
    missing = [k for k in _POLICY_KEYS if k not in policy]
    if missing:
        raise RangePolicyError(f"missing range_policy keys: {missing}")
    return (
        _check_positive_int("max_ranges", policy["max_ranges"], 1),
        _check_positive_int("max_request_bytes", policy["max_request_bytes"], 1),
        _check_positive_int("extra_bytes", policy["extra_bytes"], 0),
    )


def validate_cat_ranges_args(paths, starts, ends):
    """Validate the ``(paths, starts, ends)`` triple of a policy-driven
    ``cat_ranges`` call.

    All three must be lists of equal length; every start/end must be a
    non-bool integer with ``0 <= start <= end`` (end exclusive). Returns
    a list of ``(index, path, start, end)`` tuples. Raises before any
    request is issued, so invalid input never leaves requests behind.
    """
    if not isinstance(paths, list):
        raise TypeError(
            "paths must be a list when range_policy is used, "
            f"got {type(paths).__name__}"
        )
    if not isinstance(starts, list) or not isinstance(ends, list):
        raise RangePolicyError(
            "starts and ends must be lists when range_policy is used"
        )
    if len(paths) != len(starts) or len(paths) != len(ends):
        raise RangePolicyError(
            "paths, starts and ends must have equal lengths, got "
            f"{len(paths)}, {len(starts)} and {len(ends)}"
        )
    out = []
    for i, (p, s, e) in enumerate(zip(paths, starts, ends)):
        for name, v in (("start", s), ("end", e)):
            if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                raise RangePolicyError(
                    f"range {name} at position {i} must be an integer, got {v!r}"
                )
        s, e = int(s), int(e)
        if s < 0 or e < s:
            raise RangePolicyError(
                f"invalid range [{s}, {e}) at position {i}: require 0 <= start <= end"
            )
        out.append((i, p, s, e))
    return out


class RangePlan(NamedTuple):
    """Output of :func:`plan_range_requests`.

    requests: list of ``(url, ((start, end), ...))``; the intervals of
        each request are sorted, disjoint and sum to <= max_request_bytes.
    input_map: for each input range (in the order given to
        :func:`plan_range_requests`), the tuple of request indices whose
        declared intervals cover it, in ascending order.
    union_bytes: total length ``U`` of the union of the input ranges.
    declared_bytes: total length of all declared request intervals
        (<= union_bytes + extra_bytes).
    planner_spend: source bytes the planner itself charged to the
        budget by bridging gaps (``declared_bytes - union_bytes``).
    budget: the call-level :class:`CoalesceBudget`, shared across
        backends when a batch was planned in several pieces; its
        ``remaining`` field is the slack left for server-coalesced
        responses after the planner's own gap bridging.
    """

    requests: list
    input_map: list
    union_bytes: int
    declared_bytes: int
    planner_spend: int
    budget: "CoalesceBudget"


def _union_intervals(intervals):
    """Merge ``(start, end)`` intervals into sorted, disjoint union intervals."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _plan_url(union, max_ranges, max_request_bytes, budget, requests, segs_out, url):
    """Greedily pack one URL's union intervals into requests.

    Sweeps the sorted union intervals left to right, keeping an open
    segment. A new interval closes the segment if its bytes alone would
    exceed ``max_request_bytes``; if the segment's interval count would
    exceed ``max_ranges``, the smallest gaps in the segment are bridged
    (charged to the shared call-level ``budget``) to merge intervals.
    Appends ``(url, intervals)`` to ``requests`` and, per declared
    interval, ``(start, end, request_index)`` to ``segs_out``.
    """
    cur = []  # union intervals in the open segment
    gaps = []  # gaps[i] sits between cur[i] and cur[i + 1]
    unbridged = []  # heap of (gap, position) not yet bridged
    bridged = set()  # positions already bridged
    src_bytes = 0  # sum of union interval lengths in the open segment
    bridge_bytes = 0  # sum of bridged gap lengths in the open segment

    def emit():
        nonlocal cur, gaps, unbridged, bridged, src_bytes, bridge_bytes
        if not cur:
            return
        declared = []
        s0, e0 = cur[0]
        for i in range(1, len(cur)):
            if i - 1 in bridged:
                e0 = cur[i][1]
            else:
                declared.append((s0, e0))
                s0, e0 = cur[i]
        declared.append((s0, e0))
        idx = len(requests)
        requests.append((url, tuple(declared)))
        for s, e in declared:
            segs_out.append((s, e, idx))
        cur, gaps, unbridged, bridged = [], [], [], set()
        src_bytes = bridge_bytes = 0

    for s, e in union:
        length = e - s
        if length > max_request_bytes:
            # too long for any single request: emit full chunks directly
            # (a chunk of exactly max_request_bytes can never share a
            # request, so this does not increase the request count)
            emit()
            while e - s > max_request_bytes:
                idx = len(requests)
                requests.append((url, ((s, s + max_request_bytes),)))
                segs_out.append((s, s + max_request_bytes, idx))
                s += max_request_bytes
            length = e - s
        if not cur:
            cur = [(s, e)]
            src_bytes = length
            continue
        gap = s - cur[-1][1]
        new_src = src_bytes + length
        if new_src > max_request_bytes:
            # bridging only adds bytes, so this cannot be fixed here
            emit()
            cur = [(s, e)]
            src_bytes = length
            continue
        heapq.heappush(unbridged, (gap, len(gaps)))
        needed = len(cur) + 1 - len(bridged) - max_ranges
        if needed > 0:
            # too many intervals: bridge the cheapest gaps, if the
            # call budget and the segment byte budget allow it
            popped = []
            cost = 0
            while len(popped) < needed and unbridged:
                g2, pos = heapq.heappop(unbridged)
                popped.append((g2, pos))
                cost += g2
            if (
                len(popped) == needed
                and cost <= budget.remaining
                and new_src + bridge_bytes + cost <= max_request_bytes
            ):
                for _, pos in popped:
                    bridged.add(pos)
                bridge_bytes += cost
                budget.take(cost)
            else:
                for g2, pos in popped:
                    heapq.heappush(unbridged, (g2, pos))
                emit()
                cur = [(s, e)]
                src_bytes = length
                continue
        cur.append((s, e))
        gaps.append(gap)
        src_bytes = new_src
    emit()


def plan_range_requests(items, max_ranges, max_request_bytes, extra_bytes, budget=None):
    """Plan coalesced multi-range requests for a batch of ranges.

    Parameters
    ----------
    items: list of ``(url, start, end)`` with ``0 <= start < end``
        (zero-length ranges must be filtered by the caller; they need
        no request). URLs compare by full identity, query string
        included: only ranges on the same URL share a request.
    max_ranges, max_request_bytes, extra_bytes: see module docstring.
    budget: CoalesceBudget, optional
        A call-level budget shared with other planners and with the
        response checks. When omitted, a fresh one is created from
        ``extra_bytes``, which is the behaviour for a standalone call.
        When supplied, only its currently remaining bytes may be spent
        on gap bridging and the same object is returned in the plan, so
        several backends planning sequentially draw from one budget.

    Returns
    -------
    RangePlan plus a ``budget`` attribute carrying the (possibly shared)
    budget object whose ``remaining`` field is the slack left for
    servers that coalesce beyond the declared intervals.
    """
    if budget is None:
        budget = CoalesceBudget(extra_bytes)
    by_url = {}
    for pos, (url, s, e) in enumerate(items):
        by_url.setdefault(url, []).append((pos, s, e))

    requests = []
    segs_by_url = {}
    union_bytes = 0
    for url, members in by_url.items():
        union = _union_intervals([(s, e) for _, s, e in members])
        union_bytes += sum(e - s for s, e in union)
        segs = []
        _plan_url(union, max_ranges, max_request_bytes, budget, requests, segs, url)
        segs_by_url[url] = segs

    declared_bytes = sum(e - s for _, ivs in requests for s, e in ivs)

    # sparse input -> covering-requests map, by walking each URL's
    # sorted declared intervals (inputs never span union gaps, so the
    # walk below always finds contiguous coverage)
    input_map = [None] * len(items)
    for url, members in by_url.items():
        segs = segs_by_url[url]
        ends = [e for _, e, _ in segs]
        for pos, s, e in members:
            i = bisect.bisect_right(ends, s)
            reqs = []
            covered = s
            while covered < e:
                ss, ee, r = segs[i]
                assert ss <= covered < ee  # coverage is complete by construction
                if not reqs or reqs[-1] != r:
                    reqs.append(r)
                covered = ee
                i += 1
            input_map[pos] = tuple(reqs)

    planner_spend = declared_bytes - union_bytes
    return RangePlan(
        requests,
        input_map,
        union_bytes,
        declared_bytes,
        planner_spend,
        budget,
    )
