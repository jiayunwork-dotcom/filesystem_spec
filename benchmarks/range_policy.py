"""Comparison of the two candidate range-planning strategies, and
planning-cost scaling for the adopted one.

Candidate A ("local greedy", adopted in ``fsspec.multirange``):
    per-URL left-to-right sweep; a segment is closed when its byte
    budget is exhausted, and the cheapest gaps *inside the open
    segment* are bridged only when the interval count would otherwise
    exceed ``max_ranges``.

Candidate B ("batch-global pre-bridging"):
    per URL, first compute all gaps and bridge the globally cheapest
    ones as long as the estimated request count
    ``max(ceil(intervals/max_ranges), ceil(bytes/max_request_bytes))``
    strictly decreases and the extra budget lasts; then pack the
    merged intervals contiguously (first-fit).

Both strategies honour the same hard constraints (<= max_ranges
intervals and <= max_request_bytes per request, declared total <=
U + extra_bytes); what differs is request count, bandwidth (declared
bytes) and planning cost. Run with::

    python benchmarks/range_policy.py
"""

import bisect
import random
import sys
import time

from fsspec.multirange import _union_intervals, plan_range_requests


def _pack_first_fit(url, merged, max_ranges, max_request_bytes, requests, segs):
    """Contiguous first-fit packing of merged intervals (with splitting)."""
    cur = []
    cur_bytes = 0

    def emit():
        nonlocal cur, cur_bytes
        if cur:
            idx = len(requests)
            requests.append((url, tuple(cur)))
            segs.extend((s, e, idx) for s, e in cur)
            cur = []
            cur_bytes = 0

    for s, e in merged:
        if e - s > max_request_bytes:
            emit()
            while e - s > max_request_bytes:
                idx = len(requests)
                requests.append((url, ((s, s + max_request_bytes),)))
                segs.append((s, s + max_request_bytes, idx))
                s += max_request_bytes
        if cur and (len(cur) + 1 > max_ranges or cur_bytes + e - s > max_request_bytes):
            emit()
        cur.append((s, e))
        cur_bytes += e - s
    emit()


def plan_global_prebridge(items, max_ranges, max_request_bytes, extra_bytes):
    """Candidate B: global cheapest-gap pre-bridging + first-fit packing."""
    by_url = {}
    for pos, (url, s, e) in enumerate(items):
        by_url.setdefault(url, []).append((pos, s, e))

    requests = []
    segs_by_url = {}
    union_bytes = 0
    extra_remaining = extra_bytes
    for url, members in by_url.items():
        union = _union_intervals([(s, e) for _, s, e in members])
        union_bytes += sum(e - s for s, e in union)
        # phase 1: bridge globally cheapest gaps while the estimated
        # request count strictly decreases
        n_iv = len(union)
        total = sum(e - s for s, e in union)

        def est(n, b):
            return max(
                -(-n // max_ranges),
                -(-b // max_request_bytes),
            )

        current = est(n_iv, total)
        bridged = set()
        gaps = sorted(
            (b[0] - a[1], i) for i, (a, b) in enumerate(zip(union, union[1:]))
        )
        for g, gi in gaps:
            if g > extra_remaining:
                break
            nxt = est(n_iv - 1, total + g)
            if nxt < current:
                bridged.add(gi)
                n_iv -= 1
                total += g
                extra_remaining -= g
                current = nxt
        # apply bridging
        merged = []
        s0, e0 = union[0]
        for i in range(1, len(union)):
            if i - 1 in bridged:
                e0 = union[i][1]
            else:
                merged.append((s0, e0))
                s0, e0 = union[i]
        merged.append((s0, e0))
        # phase 2: first-fit contiguous packing (with splitting)
        segs = []
        _pack_first_fit(url, merged, max_ranges, max_request_bytes, requests, segs)
        segs_by_url[url] = segs

    declared_bytes = sum(e - s for _, ivs in requests for s, e in ivs)
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
                assert ss <= covered < ee
                if not reqs or reqs[-1] != r:
                    reqs.append(r)
                covered = ee
                i += 1
            input_map[pos] = tuple(reqs)
    return requests, input_map, union_bytes, declared_bytes


def check_invariants(requests, union_bytes, declared_bytes, policy):
    mr, mb, ex = policy
    for url, ivs in requests:
        assert len(ivs) <= mr, "too many intervals in a request"
        assert sum(e - s for s, e in ivs) <= mb, "request byte limit exceeded"
        for (s1, e1), (s2, e2) in zip(ivs, ivs[1:]):
            assert s1 < e1 <= s2 < e2, "intervals not sorted/disjoint"
    assert declared_bytes <= union_bytes + ex, "call budget exceeded"


def run_case(name, items, policy):
    mr, mb, ex = policy
    rows = []
    for label, fn in (
        ("A:local-greedy", plan_range_requests),
        ("B:global-prebridge", plan_global_prebridge),
    ):
        t0 = time.perf_counter()
        plan = fn(items, mr, mb, ex)
        dt = time.perf_counter() - t0
        requests, _, union_bytes, declared = (
            (plan.requests, plan.input_map, plan.union_bytes, plan.declared_bytes)
            if label.startswith("A")
            else plan
        )
        check_invariants(requests, union_bytes, declared, policy)
        rows.append(
            (
                label,
                len(requests),
                declared,
                declared - union_bytes,
                dt * 1e3,
            )
        )
    print(
        f"\n### {name}  (inputs={len(items)}, policy={dict(zip(('max_ranges', 'max_request_bytes', 'extra_bytes'), policy))})"
    )
    print(
        f"{'strategy':<22}{'requests':>10}{'declared B':>14}{'extra B':>10}{'plan ms':>10}"
    )
    for label, nreq, declared, extra, dt in rows:
        print(f"{label:<22}{nreq:>10}{declared:>14}{extra:>10}{dt:>10.2f}")
    return rows


def main():
    rng = random.Random(20240601)
    cases = []

    # dense small blocks: the documented benchmark shape
    cases.append(
        (
            "dense 256x64B @ stride 80",
            [("u", 80 * j, 80 * j + 64) for j in range(256)],
            (8, 4096, 4096),
        )
    )
    # sparse blocks, no extra budget
    cases.append(
        (
            "sparse 256x64B @ stride 1024",
            [("u", 1024 * j, 1024 * j + 64) for j in range(256)],
            (8, 4096, 0),
        )
    )
    # dense at scale
    cases.append(
        (
            "dense 10000x64B @ stride 80",
            [("u", 80 * j, 80 * j + 64) for j in range(10000)],
            (8, 4096, 4096),
        )
    )
    # sparse at scale over 1 GiB
    items = []
    for j in range(10000):
        off = rng.randrange(0, 2**30 - 2048)
        items.append(("u", off, off + rng.randrange(512, 1536)))
    cases.append(("sparse 10000 refs @ 1 GiB", items, (16, 65536, 65536)))

    # skewed: dense clusters far apart + sparse outliers
    items = []
    for c in range(8):
        base = c * 1_000_000
        items += [("u", base + 80 * j, base + 80 * j + 64) for j in range(128)]
    for j in range(128):
        off = rng.randrange(0, 8_000_000)
        items.append(("u", off, off + 512))
    cases.append(("skewed clusters + outliers", items, (8, 4096, 4096)))

    # overlapping references: duplicates and contained ranges
    items = []
    for j in range(1000):
        s = rng.randrange(0, 100_000)
        items.append(("u", s, s + rng.randrange(64, 2048)))
    items += items[:500]  # exact duplicates
    items += [("u", s + 10, s + 60) for _, s, _ in items[:500]]  # contained
    cases.append(("duplicates + contained", items, (8, 4096, 4096)))

    # random mixed sizes on several URLs
    items = []
    for j in range(5000):
        url = f"u{j % 5}"
        s = rng.randrange(0, 10_000_000)
        items.append((url, s, s + rng.randrange(1, 2000)))
    cases.append(("random mixed, 5 URLs", items, (8, 8192, 8192)))

    summary = {}
    for name, items, policy in cases:
        rows = run_case(name, items, policy)
        summary[name] = {r[0]: r for r in rows}

    print("\n### planning-cost scaling of the adopted strategy (A)")
    for n in (1000, 10000, 100000):
        items = []
        for j in range(n):
            off = rng.randrange(0, 2**30 - 2048)
            items.append(("u", off, off + rng.randrange(512, 1536)))
        t0 = time.perf_counter()
        plan = plan_range_requests(items, 16, 65536, 65536)
        dt = time.perf_counter() - t0
        print(
            f"n={n:<7} requests={len(plan.requests):<7} "
            f"declared={plan.declared_bytes:<10} plan={dt * 1e3:.1f}ms"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
