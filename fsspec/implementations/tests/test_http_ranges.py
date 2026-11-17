"""Tests for coalesced multi-range reads (``range_policy``).

These tests run against a local HTTP server that records the Range
headers it receives and the number of body bytes it transfers, and that
generates response bytes from its own deterministic rule (shared with
the expectations below, never with the planner under test).
"""

import asyncio
import concurrent.futures
import random
import socket
import threading
import time
import tracemalloc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

import fsspec
import fsspec.asyn
from fsspec.implementations.reference import ReferenceFileSystem, ReferenceNotReachable
from fsspec.multirange import RangeResponseError

# --------------------------------------------------------------------------
# byte generation rule shared by the test server and the expectations
# --------------------------------------------------------------------------

PATTERN = bytes((i * 131 + 17) % 256 for i in range(256))
FAKE_BOUNDARY = b"\r\n--BOUNDARY\r\n"
TRICKY_UNIT = FAKE_BOUNDARY + PATTERN[:200]


def gen(start, end):
    """Bytes of a /data file in [start, end): periodic pattern."""
    if end <= start:
        return b""
    n = end - start
    off = start % 256
    return (PATTERN[off:] + PATTERN * (n // 256 + 1))[:n]


def tricky_gen(start, end):
    """Bytes of a ?tricky=1 file: contains fake MIME boundaries."""
    if end <= start:
        return b""
    n = end - start
    off = start % len(TRICKY_UNIT)
    return (TRICKY_UNIT[off:] + TRICKY_UNIT * (n // len(TRICKY_UNIT) + 1))[:n]


def union_len(pairs):
    """Total length of the union of [s, e) intervals (test-owned helper)."""
    total = 0
    ce = cs = None
    for s, e in sorted(pairs):
        if ce is None or s > ce:
            if ce is not None:
                total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        total += ce - cs
    return total


def declared_bytes(range_header):
    """Total length of the intervals declared in a Range header."""
    if range_header is None:
        return 0
    assert range_header.startswith("bytes=")
    total = 0
    for part in range_header[6:].split(","):
        a, _, b = part.strip().partition("-")
        total += int(b) - int(a) + 1
    return total


# --------------------------------------------------------------------------
# recording HTTP test server
# --------------------------------------------------------------------------


class RangeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.log = []
        self.inflight = 0
        self.peak = 0

    def record(self, rec):
        with self.lock:
            self.log.append(rec)


def parse_ranges(header):
    out = []
    for part in header[len("bytes=") :].split(","):
        a, _, b = part.strip().partition("-")
        out.append((int(a), int(b) + 1))
    return out


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def handle(self):
            # clients may abort mid-response (e.g. on over-limit bodies)
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            with state.lock:
                state.inflight += 1
                state.peak = max(state.peak, state.inflight)
            try:
                self._handle()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with state.lock:
                    state.inflight -= 1

        def _respond(self, code, headers, body, rec, close=False):
            rec = dict(rec, status=code, sent=len(body))
            state.record(rec)
            headers = dict(headers)
            headers.setdefault("Content-Length", len(body))
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, str(v))
            self.end_headers()
            if body:
                self.wfile.write(body)
            if close:
                self.close_connection = True

        def _handle(self):
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            flag = lambda name: name in query  # noqa: E731
            path = parts.path
            if path.startswith("/alias"):
                path = path[len("/alias") :]
            rec = {
                "path": self.path,
                "range": self.headers.get("Range"),
                "accept_encoding": self.headers.get("Accept-Encoding"),
            }
            if path != "/data":
                self._respond(404, {}, b"not found", rec)
                return
            size = int(query.get("size", ["1000000"])[0])
            if "slow" in query:
                time.sleep(int(query["slow"][0]) / 1000)
            content = tricky_gen if flag("tricky") else gen
            boundary = "BOUNDARY" if flag("tricky") else "testboundary"
            range_header = rec["range"]
            ranges = parse_ranges(range_header) if range_header else None

            if flag("status416") or (ranges and any(s >= size for s, _ in ranges)):
                self._respond(
                    416,
                    {"Content-Range": f"bytes */{size}", "Content-Length": 0},
                    b"",
                    rec,
                )
                return
            if not ranges or flag("ignore"):
                body = content(0, size)
                headers = {}
                if flag("encoding"):
                    headers["Content-Encoding"] = "gzip"
                self._respond(200, headers, body, rec)
                return

            headers = {}
            if flag("encoding"):
                headers["Content-Encoding"] = "gzip"
            if flag("coalesce"):
                s0 = min(s for s, _ in ranges)
                e0 = max(e for _, e in ranges)
                body = content(s0, e0)
                headers["Content-Range"] = f"bytes {s0}-{e0 - 1}/{size}"
                self._respond(206, headers, body, rec)
                return
            if len(ranges) == 1:
                s, e = ranges[0]
                if flag("shift_range"):
                    # claim a different (shifted) range than requested
                    s += 1
                    e += 1
                body = content(s, e)
                headers["Content-Range"] = f"bytes {s}-{e - 1}/{size}"
                if flag("bad_length"):
                    headers["Content-Length"] = len(body) + 10
                if "truncate" in query:
                    n = int(query["truncate"][0])
                    headers["Content-Length"] = len(body)  # lie, then cut off
                    self._respond(206, headers, body[:n], rec, close=True)
                    return
                self._respond(206, headers, body, rec)
                return

            # multi-range request -> multipart/byteranges
            if "coalesce_parts" in query:
                k = int(query["coalesce_parts"][0])
                groups = [ranges[i : i + k] for i in range(0, len(ranges), k)]
            else:
                groups = [[r] for r in ranges]  # one part per requested range
            spans = [(min(s for s, _ in g), max(e for _, e in g)) for g in groups if g]
            if flag("huge_part"):
                # claim (and start streaming) a part far beyond the client's
                # per-response limit
                spans[0] = (0, 10_000_000)
            if flag("overlap") and len(spans) > 1:
                spans[0] = (spans[0][0], spans[1][0] + 5)
            if flag("dup"):
                spans = spans + [spans[0]]
            if flag("drop_part") and len(spans) > 1:
                spans.pop(len(spans) // 2)
            parts_out = []
            for i, (s0, e0) in enumerate(spans):
                total = size + 1 if (flag("bad_total") and i == 1) else size
                cr = f"bytes {s0}-{e0 - 1}/{total}"
                if flag("bad_range") and i == 0:
                    cr = f"bytes {e0 - 1}-{s0}/{size}"  # reversed: inconsistent
                pad = (
                    "".join(f"X-Pad-{j}: " + "x" * 4000 + "\r\n" for j in range(18))
                    if flag("pad_headers")
                    else ""
                )
                hdr = (
                    "Content-Type: application/octet-stream\r\n"
                    f"Content-Range: {cr}\r\n{pad}\r\n"
                )
                parts_out.append(hdr.encode() + content(s0, e0))
            if flag("shuffle"):
                random.Random(42).shuffle(parts_out)
            bnd = boundary.encode()
            body = b"".join(b"--" + bnd + b"\r\n" + p + b"\r\n" for p in parts_out)
            body += b"--" + bnd + b"--\r\n"
            ctype = f"multipart/byteranges; boundary={boundary}"
            if flag("quote"):
                ctype = f'multipart/byteranges; boundary="{boundary}"'
            headers["Content-Type"] = ctype
            if "truncate" in query:
                n = int(query["truncate"][0])
                headers["Content-Length"] = len(body)  # lie, then cut off
                self._respond(206, headers, body[:n], rec, close=True)
                return
            self._respond(206, headers, body, rec)

    return Handler


@pytest.fixture(scope="module")
def rserver():
    state = RangeState()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        yield SimpleNamespace(
            address=f"http://127.0.0.1:{httpd.server_port}", state=state
        )
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(autouse=True)
def _clean_state(rserver):
    rserver.state.log.clear()
    rserver.state.peak = 0
    yield


@pytest.fixture
def h():
    return fsspec.filesystem("http")


def data_url(rserver, size=1000000, **flags):
    qs = "&".join([f"size={size}"] + [f"{k}={v}" for k, v in flags.items()])
    return f"{rserver.address}/data?{qs}"


def get_log(rserver):
    return [r for r in rserver.state.log if r["status"] != 404]


def wait_for_quiet(rserver, timeout=10):
    """Wait until the server has finished handling every request.

    Cancelled client requests still complete (and are recorded) on the
    server side a little later; waiting keeps the log of subsequent
    tests clean.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with rserver.state.lock:
            if rserver.state.inflight == 0:
                return
        time.sleep(0.02)
    raise TimeoutError("server did not become quiet")


POLICY = {"max_ranges": 4, "max_request_bytes": 4096, "extra_bytes": 256}


# --------------------------------------------------------------------------
# policy and input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        "notadict",
        {"max_ranges": 4, "max_request_bytes": 100},  # missing key
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": 0, "x": 1},
        {"max_ranges": True, "max_request_bytes": 100, "extra_bytes": 0},
        {"max_ranges": 0, "max_request_bytes": 100, "extra_bytes": 0},
        {"max_ranges": 4, "max_request_bytes": -5, "extra_bytes": 0},
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": -1},
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": False},
        {"max_ranges": 1.5, "max_request_bytes": 100, "extra_bytes": 0},
    ],
)
def test_policy_validation_before_requests(h, rserver, policy):
    url = data_url(rserver, 1000)
    with pytest.raises(ValueError):
        h.cat_ranges([url], [0], [10], range_policy=policy)
    assert rserver.state.log == []


@pytest.mark.parametrize(
    "npaths,nstarts,nends",
    [
        (2, 1, 2),  # length mismatch
        (1, 2, 1),
        (1, 1, 2),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ],
)
def test_input_validation_before_requests(h, rserver, npaths, nstarts, nends):
    url = data_url(rserver, 1000)
    with pytest.raises(ValueError):
        h.cat_ranges([url] * npaths, [0] * nstarts, [1] * nends, range_policy=POLICY)
    assert rserver.state.log == []


@pytest.mark.parametrize(
    "starts,ends",
    [
        ([-1], [10]),  # negative start
        ([5], [4]),  # start > end
        ([True], [10]),  # bool
        ([0], [None]),  # non-int
        ([0.5], [10]),  # non-int
    ],
)
def test_range_validation_before_requests(h, rserver, starts, ends):
    url = data_url(rserver, 1000)
    with pytest.raises(ValueError):
        h.cat_ranges([url], starts, ends, range_policy=POLICY)
    assert rserver.state.log == []


def test_paths_not_a_list(h, rserver):
    url = data_url(rserver, 1000)
    with pytest.raises(TypeError):
        h.cat_ranges(url, [0], [10], range_policy=POLICY)
    assert rserver.state.log == []


def test_max_gap_rejected_with_policy(h, rserver):
    url = data_url(rserver, 1000)
    with pytest.raises(ValueError):
        h.cat_ranges([url], [0], [10], max_gap=10, range_policy=POLICY)
    assert rserver.state.log == []


def test_empty_and_zero_length_no_requests(h, rserver):
    assert h.cat_ranges([], [], [], range_policy=POLICY) == []
    url = data_url(rserver, 1000)
    out = h.cat_ranges([url, url], [0, 100], [0, 100], range_policy=POLICY)
    assert out == [b"", b""]
    assert rserver.state.log == []


# --------------------------------------------------------------------------
# basic coalescing behaviour and budgets
# --------------------------------------------------------------------------


def test_dense_blocks_benchmark(h, rserver):
    # 256 x 64B blocks at stride 80: the documented benchmark case
    starts = [80 * j for j in range(256)]
    ends = [s + 64 for s in starts]
    policy = {"max_ranges": 8, "max_request_bytes": 4096, "extra_bytes": 4096}
    url = data_url(rserver, 20480)
    out = h.cat_ranges([url] * 256, starts, ends, range_policy=policy)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    gets = [r for r in rserver.state.log if r["range"]]
    assert 0 < len(gets) <= 8
    declared = sum(declared_bytes(r["range"]) for r in gets)
    assert declared <= 256 * 64 + 4096
    # every request respected the per-request limits
    for r in gets:
        parts = r["range"][len("bytes=") :].split(",")
        assert len(parts) <= 8
        assert declared_bytes(r["range"]) <= 4096


def test_sparse_blocks_benchmark(h, rserver):
    # stride 1024, no extra budget: 32 requests, not one extra source byte
    starts = [1024 * j for j in range(256)]
    ends = [s + 64 for s in starts]
    policy = {"max_ranges": 8, "max_request_bytes": 4096, "extra_bytes": 0}
    url = data_url(rserver, 262144)
    out = h.cat_ranges([url] * 256, starts, ends, range_policy=policy)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    gets = [r for r in rserver.state.log if r["range"]]
    assert len(gets) <= 32
    declared = sum(declared_bytes(r["range"]) for r in gets)
    assert declared == 256 * 64


def test_duplicates_and_contained(h, rserver):
    url = data_url(rserver, 1000)
    paths = [url] * 5
    starts = [0, 0, 20, 0, 90]
    ends = [100, 100, 50, 100, 110]
    out = h.cat_ranges(paths, starts, ends, range_policy=POLICY)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    # duplicates fetched once: union is [0, 110)
    declared = sum(declared_bytes(r["range"]) for r in rserver.state.log)
    assert declared <= union_len(zip(starts, ends)) + POLICY["extra_bytes"]
    assert declared >= 110


def test_multiple_urls_and_query_identity(h, rserver):
    url1 = data_url(rserver, 1000, tag="a")
    url2 = data_url(rserver, 1000, tag="b")  # same bytes, different URL
    paths = [url1, url2, url1, url2]
    starts = [0, 10, 100, 110]
    ends = [10, 20, 110, 120]
    out = h.cat_ranges(paths, starts, ends, range_policy=POLICY)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    urls_hit = {r["path"] for r in rserver.state.log}
    assert any("tag=a" in u for u in urls_hit)
    assert any("tag=b" in u for u in urls_hit)
    # ranges on different URLs never share a request: no request mixes them
    for r in rserver.state.log:
        assert ("tag=a" in r["path"]) != ("tag=b" in r["path"])


def test_alias_urls_not_merged(h, rserver):
    url1 = data_url(rserver, 1000)
    url2 = f"{rserver.address}/alias/data?size=1000"
    starts = [0, 0]
    ends = [50, 50]
    out = h.cat_ranges([url1, url2], starts, ends, range_policy=POLICY)
    assert out == [gen(0, 50), gen(0, 50)]
    assert len([r for r in rserver.state.log if r["range"]]) == 2


def test_split_large_range(h, rserver):
    url = data_url(rserver, 30000)
    policy = {"max_ranges": 4, "max_request_bytes": 4096, "extra_bytes": 0}
    out = h.cat_ranges([url], [100], [9500], range_policy=policy)
    assert out == [gen(100, 9500)]
    gets = [r for r in rserver.state.log if r["range"]]
    assert len(gets) == 3  # 4096 + 4096 + 1208
    assert sum(declared_bytes(r["range"]) for r in gets) == 9400


def test_exact_budget_boundary(h, rserver):
    # two 40-byte blocks 60 bytes apart; bridging the gap costs exactly
    # the extra budget
    url = data_url(rserver, 1000)
    policy = {"max_ranges": 1, "max_request_bytes": 100, "extra_bytes": 20}
    out = h.cat_ranges([url, url], [0, 60], [40, 100], range_policy=policy)
    assert out == [gen(0, 40), gen(60, 100)]
    gets = [r for r in rserver.state.log if r["range"]]
    assert len(gets) == 1
    assert declared_bytes(gets[0]["range"]) == 100  # == U + extra exactly


# --------------------------------------------------------------------------
# RFC 9110 response handling
# --------------------------------------------------------------------------


def multi_policy():
    return {"max_ranges": 8, "max_request_bytes": 4096, "extra_bytes": 0}


def sparse_inputs(n=6, stride=1000, length=40, size=10000):
    starts = [stride * j for j in range(n)]
    ends = [s + length for s in starts]
    return starts, ends, size


def test_multipart_basic(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    assert len(rserver.state.log) == 1
    assert rserver.state.log[0]["range"].count(",") == 5  # one multi-range GET


def test_multipart_out_of_order(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, shuffle=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert out == [gen(s, e) for s, e in zip(starts, ends)]


def test_multipart_quoted_boundary(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, quote=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert out == [gen(s, e) for s, e in zip(starts, ends)]


def test_multipart_boundary_like_payload(h, rserver):
    # the payload itself contains perfect fake delimiters
    starts = [0, 100, 200, 300]
    ends = [90, 190, 290, 400]
    url = data_url(rserver, 1000, tricky=1)
    policy = {"max_ranges": 4, "max_request_bytes": 1000, "extra_bytes": 0}
    out = h.cat_ranges([url] * 4, starts, ends, range_policy=policy)
    assert out == [tricky_gen(s, e) for s, e in zip(starts, ends)]
    assert b"--BOUNDARY--" not in out[0]  # sanity: no parser confusion
    assert len(rserver.state.log) == 1


def test_server_coalesces_parts_accepted(h, rserver):
    # server merges adjacent parts; the new source bytes fit the budgets
    starts, ends, size = sparse_inputs(n=4, stride=100, length=40)
    policy = {"max_ranges": 4, "max_request_bytes": 4096, "extra_bytes": 200}
    url = data_url(rserver, size, coalesce_parts=2)
    out = h.cat_ranges([url] * 4, starts, ends, range_policy=policy)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]


def test_server_coalesce_single_range(h, rserver):
    starts, ends, size = sparse_inputs(n=3, stride=100, length=40)
    policy = {"max_ranges": 3, "max_request_bytes": 300, "extra_bytes": 200}
    url = data_url(rserver, size, coalesce=1)
    out = h.cat_ranges([url] * 3, starts, ends, range_policy=policy)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    assert len(rserver.state.log) == 1


def test_server_coalesce_exactly_at_budget(h, rserver):
    # coalesced response uses exactly the remaining call budget
    policy = {"max_ranges": 2, "max_request_bytes": 100, "extra_bytes": 20}
    url = data_url(rserver, 1000, coalesce=1)
    out = h.cat_ranges([url, url], [0, 60], [40, 100], range_policy=policy)
    assert out == [gen(0, 40), gen(60, 100)]


def test_server_coalesce_over_budget(h, rserver):
    policy = {"max_ranges": 2, "max_request_bytes": 100, "extra_bytes": 19}
    url = data_url(rserver, 1000, coalesce=1)
    out = h.cat_ranges([url, url], [0, 60], [40, 100], range_policy=policy)
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_server_coalesce_over_max_request_bytes(h, rserver):
    policy = {"max_ranges": 2, "max_request_bytes": 99, "extra_bytes": 100}
    url = data_url(rserver, 1000, coalesce=1)
    out = h.cat_ranges([url, url], [0, 60], [40, 100], range_policy=policy)
    assert all(isinstance(o, RangeResponseError) for o in out)


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_ignored_range_200_fails(h, rserver):
    url = data_url(rserver, 1000, ignore=1)
    out = h.cat_ranges([url], [10], [50], range_policy=POLICY)
    assert isinstance(out[0], RangeResponseError)
    # the whole file was not downloaded to satisfy the range
    assert rserver.state.log[0]["sent"] <= 1000


def test_416_fails(h, rserver):
    url = data_url(rserver, 1000, status416=1)
    out = h.cat_ranges([url], [10], [50], range_policy=POLICY)
    assert isinstance(out[0], RangeResponseError)


def test_truncated_body_fails(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, truncate=30)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, Exception) for o in out)


def test_truncated_single_range_fails(h, rserver):
    url = data_url(rserver, 1000, truncate=10)
    out = h.cat_ranges([url], [0], [100], range_policy=POLICY)
    assert isinstance(out[0], Exception)


def test_missing_part_fails(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, drop_part=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_overlapping_parts_fail(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, overlap=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_duplicate_part_fails(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, dup=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_conflicting_totals_fail(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, bad_total=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_bad_part_range_fails(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, bad_range=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_single_range_mismatch_fails(h, rserver):
    url = data_url(rserver, 1000, shift_range=1)
    out = h.cat_ranges([url], [10], [50], range_policy=POLICY)
    assert isinstance(out[0], RangeResponseError)


def test_content_length_mismatch_fails(h, rserver):
    url = data_url(rserver, 1000, bad_length=1)
    out = h.cat_ranges([url], [10], [50], range_policy=POLICY)
    assert isinstance(out[0], RangeResponseError)


def test_requests_use_identity_encoding(h, rserver):
    url = data_url(rserver, 1000)
    out = h.cat_ranges([url], [10], [50], range_policy=POLICY)
    assert out == [gen(10, 50)]
    assert rserver.state.log[0]["accept_encoding"] == "identity"


def test_content_encoding_rejected(h, rserver):
    url = data_url(rserver, 1000, encoding="gzip")
    out = h.cat_ranges([url], [0], [50], range_policy=POLICY)
    assert isinstance(out[0], RangeResponseError)


def test_oversized_part_aborts_bounded(h, rserver):
    # a part claiming 10 MB must be aborted, not buffered: the policy
    # allows only 4096 source bytes per response
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, huge_part=1)
    t0 = time.time()
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)
    assert time.time() - t0 < 10  # did not download the 10 MB first


def test_mime_overhead_bound(h, rserver):
    # 70 KB of MIME part headers exceeds the 64 KiB framing allowance
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size, pad_headers=1)
    out = h.cat_ranges([url] * 6, starts, ends, range_policy=multi_policy())
    assert all(isinstance(o, RangeResponseError) for o in out)


def test_network_error_fails(h):
    # bind then release a port to guarantee nothing listens on it
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}/data?size=1000"
    out = h.cat_ranges([url], [0], [50], range_policy=POLICY)
    assert isinstance(out[0], OSError)
    with pytest.raises(OSError):
        h.cat_ranges([url], [0], [50], range_policy=POLICY, on_error="raise")


def test_on_error_raise(h, rserver):
    url = data_url(rserver, 1000, status416=1)
    with pytest.raises(RangeResponseError):
        h.cat_ranges([url], [0], [50], range_policy=POLICY, on_error="raise")


def test_on_error_bad_value(h, rserver):
    url = data_url(rserver, 1000)
    with pytest.raises(ValueError):
        h.cat_ranges([url], [0], [50], range_policy=POLICY, on_error="omit")


def test_one_request_fails_others_succeed(h, rserver):
    bad = data_url(rserver, 1000, status416=1, tag="bad")
    good = data_url(rserver, 1000, tag="good")
    paths = [bad, good, bad, good]
    starts = [0, 0, 10, 10]
    ends = [10, 10, 20, 20]
    out = h.cat_ranges(paths, starts, ends, range_policy=POLICY)
    assert isinstance(out[0], RangeResponseError)
    assert out[1] == gen(0, 10)
    assert isinstance(out[2], RangeResponseError)
    assert out[3] == gen(10, 20)


def test_404_raises_filenotfound(h, rserver):
    url = f"{rserver.address}/missing"
    out = h.cat_ranges([url], [0], [50], range_policy=POLICY)
    assert isinstance(out[0], FileNotFoundError)


# --------------------------------------------------------------------------
# async entry points, batching and cancellation
# --------------------------------------------------------------------------


def test_async_cat_ranges_direct(h, rserver):
    starts, ends, size = sparse_inputs()
    url = data_url(rserver, size)
    out = fsspec.asyn.sync(
        h.loop, h._cat_ranges, [url] * 6, starts, ends, range_policy=multi_policy()
    )
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    assert len(rserver.state.log) == 1


def test_batch_size_limits_inflight(rserver):
    h = fsspec.filesystem("http", batch_size=2, skip_instance_cache=True)
    starts = [400 * j for j in range(16)]
    ends = [s + 50 for s in starts]
    policy = {"max_ranges": 1, "max_request_bytes": 100, "extra_bytes": 0}
    url = data_url(rserver, 100000, slow=150)
    out = h.cat_ranges([url] * 16, starts, ends, range_policy=policy)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]
    assert len(rserver.state.log) == 16
    assert 0 < rserver.state.peak <= 2


def test_async_cancel_then_read_again(h, rserver):
    starts = [400 * j for j in range(8)]
    ends = [s + 50 for s in starts]
    policy = {"max_ranges": 1, "max_request_bytes": 100, "extra_bytes": 0}
    url = data_url(rserver, 100000, slow=400)
    coro = h._cat_ranges([url] * 8, starts, ends, range_policy=policy)
    fut = asyncio.run_coroutine_threadsafe(coro, h.loop)
    time.sleep(0.2)  # let the first requests get going
    fut.cancel()
    with pytest.raises(concurrent.futures.CancelledError):
        fut.result(timeout=10)
    wait_for_quiet(rserver)
    # the same filesystem instance keeps working afterwards
    url2 = data_url(rserver, 100000)
    out = h.cat_ranges([url2] * 8, starts, ends, range_policy=policy)
    assert out == [gen(s, e) for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------
# old behaviour without a policy
# --------------------------------------------------------------------------


def test_no_policy_unchanged(h, rserver):
    url = data_url(rserver, 1000)
    out = h.cat_ranges([url, url], [0, 100], [10, 110])
    assert out == [gen(0, 10), gen(100, 110)]
    # one single-range GET per input, as before
    gets = [r for r in rserver.state.log if r["range"]]
    assert len(gets) == 2
    assert all("," not in r["range"] for r in gets)


# --------------------------------------------------------------------------
# ReferenceFileSystem integration
# --------------------------------------------------------------------------


def make_ref_fs(rserver, refs, policy=None, **kw):
    kw.setdefault("fs", fsspec.filesystem("http"))
    return ReferenceFileSystem(refs, range_policy=policy, **kw)


def test_ref_constructor_validates_policy(rserver):
    with pytest.raises(ValueError):
        make_ref_fs(rserver, {}, policy={"max_ranges": 1})
    with pytest.raises(ValueError):
        make_ref_fs(
            rserver,
            {},
            policy={"max_ranges": 1, "max_request_bytes": 2, "extra_bytes": 0, "zz": 1},
        )


def test_ref_cat_mixed_inline_and_http(h, rserver):
    url = data_url(rserver, 10000)
    refs = {
        "inline": b"hello world",
        "a": (url, 0, 64),
        "b": (url, 64, 64),
        "c": (url, 128, 64),
    }
    fs = make_ref_fs(rserver, refs, policy=POLICY)
    out = fs.cat(["inline", "a", "b", "c"])
    assert out["inline"] == b"hello world"
    assert out["a"] == gen(0, 64)
    assert out["b"] == gen(64, 128)
    assert out["c"] == gen(128, 192)
    gets = [r for r in rserver.state.log if r["range"]]
    assert len(gets) == 1  # the three http refs coalesced into one GET


def test_ref_cat_single_key(h, rserver):
    url = data_url(rserver, 10000)
    fs = make_ref_fs(rserver, {"a": (url, 10, 50)}, policy=POLICY)
    assert fs.cat("a") == gen(10, 60)


def test_ref_cat_ranges_logical_offsets(h, rserver):
    # starts/ends are relative to each key's own length
    url = data_url(rserver, 10000)
    refs = {
        "a": (url, 1000, 100),
        "b": (url, 2000, 100),
        "inline": b"0123456789",
    }
    fs = make_ref_fs(rserver, refs, policy=POLICY)
    out = fs.cat_ranges(["a", "b", "inline", "a"], [10, 0, 2, 90], [20, 50, 5, 100])
    assert out == [gen(1010, 1020), gen(2000, 2050), b"234", gen(1090, 1100)]
    gets = [r for r in rserver.state.log if r["range"]]
    assert len(gets) == 1  # inline produced no GET; a/b coalesced


def test_ref_shared_bytes_and_contained(h, rserver):
    url = data_url(rserver, 10000)
    refs = {
        "x": (url, 500, 100),  # physical [500, 600)
        "y": (url, 550, 100),  # overlaps x
        "z": (url, 520, 20),  # contained in x
    }
    fs = make_ref_fs(rserver, refs, policy=POLICY)
    out = fs.cat(["x", "y", "z"])
    assert out["x"] == gen(500, 600)
    assert out["y"] == gen(550, 650)
    assert out["z"] == gen(520, 540)
    declared = sum(declared_bytes(r["range"]) for r in rserver.state.log)
    assert declared <= 150 + POLICY["extra_bytes"]


def test_ref_key_split_across_requests(h, rserver):
    url = data_url(rserver, 30000)
    policy = {"max_ranges": 4, "max_request_bytes": 4096, "extra_bytes": 0}
    fs = make_ref_fs(rserver, {"big": (url, 100, 9400)}, policy=policy)
    assert fs.cat("big") == gen(100, 9500)
    assert len([r for r in rserver.state.log if r["range"]]) == 3


def test_ref_inline_only_no_requests(h, rserver):
    fs = make_ref_fs(rserver, {"a": b"data", "b": b"base64:aGVsbG8="}, policy=POLICY)
    assert fs.cat(["a", "b"]) == {"a": b"data", "b": b"hello"}
    assert fs.cat_ranges(["a", "b"], [1, 0], [3, 2]) == [b"at", b"he"]
    assert rserver.state.log == []


def test_ref_zero_length_no_requests(h, rserver):
    url = data_url(rserver, 10000)
    fs = make_ref_fs(rserver, {"a": (url, 0, 100)}, policy=POLICY)
    assert fs.cat_ranges(["a", "a"], [0, 50], [0, 50]) == [b"", b""]
    assert rserver.state.log == []


def test_ref_whole_file_rejected(h, rserver):
    url = data_url(rserver, 10000)
    fs = make_ref_fs(rserver, {"whole": [url], "a": (url, 0, 10)}, policy=POLICY)
    with pytest.raises(ValueError):
        fs.cat(["whole", "a"])
    with pytest.raises(ValueError):
        fs.cat_ranges(["whole"], [0], [1])
    assert rserver.state.log == []


def test_ref_other_protocol_rejected(rserver, m):
    m.pipe("/data/0", b"x" * 100)
    url = data_url(rserver, 10000)
    fs = make_ref_fs(
        rserver,
        {"mem": ("memory://data/0", 0, 10), "a": (url, 0, 10)},
        policy=POLICY,
        fs={"memory": m, "http": fsspec.filesystem("http")},
    )
    with pytest.raises(ValueError):
        fs.cat(["mem", "a"])
    with pytest.raises(ValueError):
        fs.cat_ranges(["mem"], [0], [5])
    assert rserver.state.log == []


def test_ref_out_of_bounds_rejected(h, rserver):
    url = data_url(rserver, 10000)
    fs = make_ref_fs(rserver, {"a": (url, 0, 100), "i": b"abc"}, policy=POLICY)
    with pytest.raises(ValueError):
        fs.cat_ranges(["a"], [0], [101])
    with pytest.raises(ValueError):
        fs.cat_ranges(["i"], [0], [4])
    assert rserver.state.log == []


def test_ref_cat_on_error_modes(h, rserver):
    bad = data_url(rserver, 1000, status416=1, tag="bad")
    good = data_url(rserver, 1000, tag="good")
    refs = {"bad1": (bad, 0, 10), "bad2": (bad, 10, 10), "ok": (good, 0, 10)}
    fs = make_ref_fs(rserver, refs, policy=POLICY)
    # raise
    with pytest.raises(ReferenceNotReachable) as exc:
        fs.cat(["bad1", "bad2", "ok"], on_error="raise")
    assert exc.value.reference == "bad1"
    # return
    out = fs.cat(["bad1", "bad2", "ok"], on_error="return")
    assert isinstance(out["bad1"], ReferenceNotReachable)
    assert isinstance(out["bad2"], ReferenceNotReachable)
    assert out["ok"] == gen(0, 10)
    # omit
    out = fs.cat(["bad1", "bad2", "ok"], on_error="omit")
    assert list(out) == ["ok"]
    assert out["ok"] == gen(0, 10)


def test_ref_cat_ranges_on_error(h, rserver):
    bad = data_url(rserver, 1000, status416=1, tag="bad")
    good = data_url(rserver, 1000, tag="good")
    refs = {"bad": (bad, 0, 20), "ok": (good, 0, 20)}
    fs = make_ref_fs(rserver, refs, policy=POLICY)
    out = fs.cat_ranges(["bad", "ok", "bad"], [0, 0, 10], [10, 10, 20])
    assert isinstance(out[0], ReferenceNotReachable)
    assert out[1] == gen(0, 10)
    assert isinstance(out[2], ReferenceNotReachable)
    with pytest.raises(ReferenceNotReachable):
        fs.cat_ranges(["bad", "ok"], [0, 0], [10, 10], on_error="raise")


def test_ref_one_failed_request_fails_only_dependents(h, rserver):
    # two keys share one failing physical file; a third key is fine
    bad = data_url(rserver, 1000, status416=1, tag="bad")
    good = data_url(rserver, 1000, tag="good")
    refs = {
        "k1": (bad, 0, 10),
        "k2": (bad, 500, 10),
        "k3": (good, 0, 10),
    }
    fs = make_ref_fs(rserver, refs, policy=POLICY)
    out = fs.cat(["k1", "k2", "k3"], on_error="return")
    assert isinstance(out["k1"], ReferenceNotReachable)
    assert isinstance(out["k2"], ReferenceNotReachable)
    assert out["k3"] == gen(0, 10)


def test_ref_missing_keys(h, rserver):
    url = data_url(rserver, 1000)
    fs = make_ref_fs(rserver, {"a": (url, 0, 10)}, policy=POLICY)
    with pytest.raises(FileNotFoundError):
        fs.cat(["nope", "a"])
    out = fs.cat(["nope", "a"], on_error="return")
    assert isinstance(out["nope"], FileNotFoundError)
    assert out["a"] == gen(0, 10)
    out = fs.cat(["nope", "a"], on_error="omit")
    assert list(out) == ["a"]
    # cat_ranges: missing key reported in place, valid key still fetched
    out = fs.cat_ranges(["nope", "a"], [0, 0], [5, 5])
    assert isinstance(out[0], FileNotFoundError)
    assert out[1] == gen(0, 5)
    with pytest.raises(FileNotFoundError):
        fs.cat_ranges(["nope", "a"], [0, 0], [5, 5], on_error="raise")


def test_ref_no_policy_regression(h, rserver):
    # without a policy, merging follows max_gap/max_block as before
    url = data_url(rserver, 10000)
    refs = {"a": (url, 0, 64), "b": (url, 64, 64)}
    fs = make_ref_fs(rserver, refs, policy=None)
    assert fs.cat(["a", "b"]) == {"a": gen(0, 64), "b": gen(64, 128)}
    # cat_ranges falls back to per-range cat_file behaviour
    out = fs.cat_ranges(["a", "b"], [0, 0], [10, 10])
    assert out == [gen(0, 10), gen(64, 74)]


def test_ref_via_public_constructor(rserver):
    url = data_url(rserver, 10000)
    refs = {"a": (url, 0, 64), "b": (url, 64, 64), "i": b"inline"}
    fs = fsspec.filesystem(
        "reference",
        fo=refs,
        remote_protocol="http",
        range_policy=POLICY,
    )
    out = fs.cat(["a", "b", "i"])
    assert out == {"a": gen(0, 64), "b": gen(64, 128), "i": b"inline"}
    assert len([r for r in rserver.state.log if r["range"]]) == 1
    out = fs.cat_ranges(["a", "b"], [0, 10], [10, 20])
    assert out == [gen(0, 10), gen(74, 84)]


# --------------------------------------------------------------------------
# scale, planning cost and memory
# --------------------------------------------------------------------------


def _make_scatter_refs(url, n, size, seed=1234):
    rng = random.Random(seed)
    refs = {}
    for j in range(n):
        off = rng.randrange(0, size - 2048)
        ln = rng.randrange(512, 1536)
        refs[f"k{j}"] = (url, off, ln)
    return refs


@pytest.mark.parametrize("n", [1000, 10000])
def test_scale_planning_and_memory(h, rserver, n):
    size = 2**30  # 1 GiB logical source, generated on demand by the server
    url = data_url(rserver, size)
    refs = _make_scatter_refs(url, n, size)
    policy = {"max_ranges": 16, "max_request_bytes": 65536, "extra_bytes": 65536}
    h2 = fsspec.filesystem("http", batch_size=32, skip_instance_cache=True)
    fs = ReferenceFileSystem(refs, fs=h2, range_policy=policy)

    # pure planning cost, measured on the same resolved ranges
    from fsspec.multirange import plan_range_requests

    resolved = [(u, off, off + ln) for u, off, ln in refs.values()]
    t0 = time.perf_counter()
    plan = plan_range_requests(resolved, 16, 65536, 65536)
    plan_dt = time.perf_counter() - t0

    tracemalloc.start()
    t0 = time.perf_counter()
    out = fs.cat(list(refs))
    total_dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert len(out) == n
    for k, v in out.items():
        u, off, ln = refs[k]
        assert v == gen(off, off + ln)

    gets = [r for r in rserver.state.log if r["range"]]
    declared = sum(declared_bytes(r["range"]) for r in gets)
    transferred = sum(r["sent"] for r in gets)
    u_len = union_len([(off, off + ln) for _, off, ln in refs.values()])
    returned = sum(ln for _, _, ln in refs.values())
    print(
        f"\nn={n}: plan={plan_dt * 1e3:.1f}ms total={total_dt:.2f}s "
        f"requests={len(gets)} declared={declared} U={u_len} "
        f"transferred={transferred} inflight_peak={rserver.state.peak} "
        f"py_peak={peak / 2**20:.1f}MiB"
    )
    assert len(gets) == len(plan.requests)
    assert declared <= u_len + policy["extra_bytes"]
    assert rserver.state.peak <= 32
    # buffers grow with returned data and batch_size*max_request_bytes,
    # not with the 1 GiB source
    assert peak < returned + 32 * 65536 + 64 * 2**20
    assert peak < size // 4
    assert plan_dt < 5
