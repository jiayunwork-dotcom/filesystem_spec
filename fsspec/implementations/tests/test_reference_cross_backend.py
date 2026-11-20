"""Cross-backend batch boundary regressions for ``range_policy``.

These tests pin the behaviour that one :meth:`ReferenceFileSystem.cat`
or :meth:`ReferenceFileSystem.cat_ranges` call is a single budget and
static-validation boundary even when its references resolve to
independent HTTPFileSystem instances for different protocols (plain
HTTP and real TLS HTTPS), each with its own headers/session/options.

A recording server (one HTTP listener, one TLS listener using the
repository's test certificate) logs every GET, its Range header and the
per-instance request header, so the tests check what was *declared* and
*accepted*, not only the returned bytes.
"""

import asyncio
import concurrent.futures
import os
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

import fsspec
import fsspec.asyn
from fsspec.implementations.reference import (
    ReferenceFileSystem,
    ReferenceNotReachable,
)
from fsspec.multirange import CoalesceBudget, RangePolicyError, RangeResponseError

CERT = os.path.join(os.path.dirname(__file__), "keycert.pem")

PATTERN = bytes((i * 131 + 17) % 256 for i in range(256))
FILE_SIZE = 1000


def gen(start, end):
    if end <= start:
        return b""
    n = end - start
    off = start % 256
    return (PATTERN[off:] + PATTERN * (n // 256 + 1))[:n]


def union_len(pairs):
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
    if range_header is None:
        return 0
    total = 0
    for part in range_header[len("bytes=") :].split(","):
        a, _, b = part.strip().partition("-")
        total += int(b) - int(a) + 1
    return total


def parse_ranges(header):
    return [
        (int(a), int(b) + 1)
        for part in header[len("bytes=") :].split(",")
        for a, _, b in [part.strip().partition("-")]
    ]


class ServerState:
    def __init__(self, scheme):
        self.scheme = scheme
        self.lock = threading.Lock()
        self.log = []
        self.inflight = 0
        self.peak = 0
        self.fail_paths = set()
        self.slow_ms = 0
        # query flag -> server answers a multi-range request by merging
        # every requested interval into one Content-Range covering range
        self.coalesce_paths = set()

    def record(self, rec):
        with self.lock:
            self.log.append(rec)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def handle(self):
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

        def _send(self, code, headers, body, rec):
            state.record(dict(rec, status=code, sent=len(body)))
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, str(v))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _handle(self):
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            tag = query.get("tag", [""])[0]
            size = int(query.get("size", [FILE_SIZE])[0])
            rec = {
                "scheme": state.scheme,
                "path": self.path,
                "range": self.headers.get("Range"),
                "tag": self.headers.get("X-Tag"),
            }
            if state.slow_ms:
                time.sleep(state.slow_ms / 1000)
            if tag in state.fail_paths:
                self._send(500, {}, b"boom", rec)
                return
            if parts.path != "/data":
                self._send(404, {}, b"not found", rec)
                return
            rng = rec["range"]
            ranges = parse_ranges(rng) if rng else None
            if not ranges:
                self._send(200, {}, gen(0, size), rec)
                return
            if any(s >= size for s, _ in ranges):
                self._send(
                    416,
                    {"Content-Range": f"bytes */{size}"},
                    b"",
                    rec,
                )
                return
            if tag in state.coalesce_paths:
                s0 = min(s for s, _ in ranges)
                e0 = max(e for _, e in ranges)
                body = gen(s0, e0)
                self._send(
                    206,
                    {"Content-Range": f"bytes {s0}-{e0 - 1}/{size}"},
                    body,
                    rec,
                )
                return
            if len(ranges) == 1:
                s, e = ranges[0]
                body = gen(s, e)
                self._send(
                    206,
                    {"Content-Range": f"bytes {s}-{e - 1}/{size}"},
                    body,
                    rec,
                )
                return
            body = b""
            for s, e in ranges:
                body += b"--testboundary\r\n"
                body += (
                    f"Content-Type: application/octet-stream\r\n"
                    f"Content-Range: bytes {s}-{e - 1}/{size}\r\n\r\n"
                ).encode()
                body += gen(s, e) + b"\r\n"
            body += b"--testboundary--\r\n"
            self._send(
                206,
                {"Content-Type": "multipart/byteranges; boundary=testboundary"},
                body,
                rec,
            )

    return Handler


def _start_server(use_ssl):
    state = ServerState("https" if use_ssl else "http")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    httpd.daemon_threads = True
    if use_ssl:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


@pytest.fixture(scope="module")
def servers():
    httpd, http_state = _start_server(False)
    httpsd, https_state = _start_server(True)
    try:
        yield SimpleNamespace(
            http=SimpleNamespace(
                address=f"http://127.0.0.1:{httpd.server_port}",
                state=http_state,
            ),
            https=SimpleNamespace(
                address=f"https://127.0.0.1:{httpsd.server_port}",
                state=https_state,
            ),
        )
    finally:
        httpd.shutdown()
        httpsd.shutdown()
        httpd.server_close()
        httpsd.server_close()


@pytest.fixture(autouse=True)
def _clear_logs(servers):
    for s in (servers.http.state, servers.https.state):
        s.log.clear()
        s.peak = 0
        s.fail_paths.clear()
        s.coalesce_paths.clear()
        s.slow_ms = 0
    yield


def url(servers, scheme, tag, size=None):
    base = getattr(servers, scheme).address
    qs = f"tag={tag}"
    if size is not None:
        qs += f"&size={size}"
    return f"{base}/data?{qs}"


def all_records(servers):
    return servers.http.state.log + servers.https.state.log


def wait_for_quiet(servers, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(s.inflight == 0 for s in (servers.http.state, servers.https.state)):
            return
        time.sleep(0.02)
    raise TimeoutError("server did not become quiet")


def make_backends():
    """Two independent HTTPFileSystem instances with distinct headers."""
    http_fs = fsspec.filesystem(
        "http", skip_instance_cache=True, headers={"X-Tag": "http-tag"}
    )
    https_fs = fsspec.filesystem(
        "https",
        skip_instance_cache=True,
        headers={"X-Tag": "https-tag"},
        ssl=False,
    )
    return {"http": http_fs, "https": https_fs}


def make_ref_fs(servers, refs, policy, backends=None):
    return ReferenceFileSystem(
        refs,
        fs=backends or make_backends(),
        range_policy=policy,
    )


def four_refs(servers):
    h = url(servers, "http", "h")
    s = url(servers, "https", "s")
    return {
        "h0": (h, 0, 1),
        "h2": (h, 2, 1),
        "s0": (s, 0, 1),
        "s2": (s, 2, 1),
    }


POLICY = {"max_ranges": 1, "max_request_bytes": 3, "extra_bytes": 1}


# --------------------------------------------------------------------------
# the headline failures: one budget across two independent backends
# --------------------------------------------------------------------------


def test_single_extra_budget_across_backends_cat(servers):
    refs = four_refs(servers)
    fs = make_ref_fs(servers, refs, POLICY)
    out = fs.cat(["h0", "h2", "s0", "s2"])
    assert out == {
        "h0": gen(0, 1),
        "h2": gen(2, 3),
        "s0": gen(0, 1),
        "s2": gen(2, 3),
    }
    gets = [r for r in all_records(servers) if r["range"]]
    # U = 4, one extra byte for the whole call: exactly three GETs,
    # declared source intervals summing to at most U + 1 = 5
    assert len(gets) == 3
    assert sum(declared_bytes(r["range"]) for r in gets) <= 5
    # both independent backends really were used, each with its header
    schemes = {r["scheme"] for r in gets}
    assert schemes == {"http", "https"}
    for r in gets:
        assert r["tag"] == f"{r['scheme']}-tag"


def test_single_extra_budget_across_backends_cat_ranges(servers):
    refs = four_refs(servers)
    fs = make_ref_fs(servers, refs, POLICY)
    keys = ["h0", "h2", "s0", "s2"]
    out = fs.cat_ranges(keys, [0] * 4, [1] * 4)
    assert out == [gen(0, 1), gen(2, 3), gen(0, 1), gen(2, 3)]
    gets = [r for r in all_records(servers) if r["range"]]
    assert len(gets) == 3
    assert sum(declared_bytes(r["range"]) for r in gets) <= 5


def test_zero_extra_budget_uses_four_gets(servers):
    refs = four_refs(servers)
    policy = {"max_ranges": 1, "max_request_bytes": 3, "extra_bytes": 0}
    fs = make_ref_fs(servers, refs, policy)
    fs.cat(["h0", "h2", "s0", "s2"])
    gets = [r for r in all_records(servers) if r["range"]]
    assert len(gets) == 4
    # no extra source byte anywhere: declared == U == 4
    assert sum(declared_bytes(r["range"]) for r in gets) == 4


def test_union_computed_per_full_url_including_query(servers):
    # same path/port but different query strings must not share requests;
    # the union U is computed per full URL then summed
    h1 = url(servers, "http", "one")
    h2 = url(servers, "http", "two")
    refs = {
        "a": (h1, 0, 10),
        "b": (h1, 5, 10),  # overlaps a on URL one: union [0, 15)
        "c": (h2, 0, 10),  # different URL, even though the bytes match
    }
    policy = {"max_ranges": 8, "max_request_bytes": 100, "extra_bytes": 0}
    fs = make_ref_fs(servers, refs, policy)
    out = fs.cat(["a", "b", "c"])
    assert out["a"] == gen(0, 10)
    assert out["b"] == gen(5, 15)
    assert out["c"] == gen(0, 10)
    gets = servers.http.state.log
    assert len(gets) == 2
    for r in gets:
        assert "tag=one" in r["path"] or "tag=two" in r["path"]
    # U = 15 + 10 = 25, no overlap merging across URLs
    assert sum(declared_bytes(r["range"]) for r in gets) == 25


# --------------------------------------------------------------------------
# server-expanded responses share the same budget as the planner
# --------------------------------------------------------------------------


def test_server_coalescing_accepted_within_shared_budget(servers):
    # only the HTTP backend coalesces; its one extra byte is the whole
    # call's extra budget, the HTTPS backend stays at its union bytes
    servers.http.state.coalesce_paths.add("h")
    refs = four_refs(servers)
    policy = {"max_ranges": 2, "max_request_bytes": 3, "extra_bytes": 1}
    fs = make_ref_fs(servers, refs, policy)
    out = fs.cat(["h0", "h2", "s0", "s2"], on_error="return")
    assert out["h0"] == gen(0, 1)
    assert out["h2"] == gen(2, 3)
    assert out["s0"] == gen(0, 1)
    assert out["s2"] == gen(2, 3)
    gets = all_records(servers)
    assert sum(declared_bytes(r["range"]) for r in gets) <= 5
    http_gets = [r for r in gets if r["scheme"] == "http"]
    assert len(http_gets) == 1
    # the request declared the two one-byte intervals; the server's
    # covering response was accepted within the remaining slack
    assert declared_bytes(http_gets[0]["range"]) == 2
    # same accounting through cat_ranges, positions aligned
    positioned = fs.cat_ranges(["h0", "h2", "s0", "s2"], [0] * 4, [1] * 4)
    assert positioned == [gen(0, 1), gen(2, 3), gen(0, 1), gen(2, 3)]


def test_server_coalescing_exhausting_budget_fails_second_backend(servers):
    # both backends would need the same single extra byte to accept the
    # covering response; the second one must be rejected, and only the
    # logical keys depending on it fail
    servers.http.state.coalesce_paths.add("h")
    servers.https.state.coalesce_paths.add("s")
    refs = four_refs(servers)
    policy = {"max_ranges": 2, "max_request_bytes": 3, "extra_bytes": 1}
    fs = make_ref_fs(servers, refs, policy)
    out = fs.cat(["h0", "h2", "s0", "s2"], on_error="return")
    assert out["h0"] == gen(0, 1)
    assert out["h2"] == gen(2, 3)
    assert isinstance(out["s0"], ReferenceNotReachable)
    assert isinstance(out["s2"], ReferenceNotReachable)
    assert isinstance(out["s0"].__cause__, RangeResponseError)
    # raise mode surfaces the same failure as ReferenceNotReachable
    with pytest.raises(ReferenceNotReachable):
        fs.cat(["h0", "h2", "s0", "s2"], on_error="raise")


def test_no_whole_file_fallback_on_over_budget(servers):
    # an over-budget coalesced response must never degrade into a 200
    # full-file fetch or a retry: every request stays ranged, and one
    # backend's keys fail (whichever is planned second), never silently
    servers.http.state.coalesce_paths.add("h")
    servers.https.state.coalesce_paths.add("s")
    refs = four_refs(servers)
    # max_ranges=2 makes both servers receive a two-interval request and
    # answer with a covering 3-byte range; the call can afford only one
    policy = {"max_ranges": 2, "max_request_bytes": 3, "extra_bytes": 1}
    fs = make_ref_fs(servers, refs, policy)
    out = fs.cat(["h0", "h2", "s0", "s2"], on_error="return")
    failures = [k for k, v in out.items() if isinstance(v, Exception)]
    assert failures  # the second backend's response was refused
    assert set(failures) in ({"h0", "h2"}, {"s0", "s2"})
    for r in all_records(servers):
        assert r["range"] is not None
        assert r["status"] != 200
    # exactly one GET per backend, no retry traffic appeared
    assert len(all_records(servers)) == 2


# --------------------------------------------------------------------------
# static validation before any GET, regardless of ordering or on_error
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [["ok", "bad"], ["bad", "ok"]])
@pytest.mark.parametrize(
    "bad_scheme,good_scheme", [("https", "http"), ("http", "https")]
)
def test_invalid_reference_fails_before_any_get_cat(
    servers, order, bad_scheme, good_scheme
):
    good_url = url(servers, good_scheme, "g")
    bad_url = url(servers, bad_scheme, "b")
    refs = {"ok": (good_url, 0, 1), "bad": (bad_url, -1, 1)}
    fs = make_ref_fs(servers, refs, POLICY)
    for mode in ("raise", "return", "omit"):
        with pytest.raises(RangePolicyError):
            fs.cat(list(order), on_error=mode)
        assert all_records(servers) == []


@pytest.mark.parametrize("order", [["ok", "bad"], ["bad", "ok"]])
def test_invalid_reference_each_protocol(servers, order):
    # negative offset on either backend, with the valid reference on the
    # other one: still a preflight failure, never a GET
    h = url(servers, "http", "h")
    s = url(servers, "https", "s")
    variants = [
        {"ok": (h, 0, 1), "bad": (s, -1, 1)},
        {"ok": (s, 0, 1), "bad": (h, -1, 1)},
        {"ok": (h, 0, 1), "bad": (s, 0, -1)},  # negative size
    ]
    for refs in variants:
        fs = make_ref_fs(servers, refs, POLICY)
        with pytest.raises(RangePolicyError):
            fs.cat(list(order))
        assert all_records(servers) == []


def test_invalid_reference_fails_before_any_get_cat_ranges(servers):
    h = url(servers, "http", "h")
    s = url(servers, "https", "s")
    refs = {"ok": (h, 0, 10), "bad": (s, -1, 10)}
    fs = make_ref_fs(servers, refs, POLICY)
    with pytest.raises(RangePolicyError):
        fs.cat_ranges(["ok", "bad"], [0, 0], [1, 1])
    with pytest.raises(RangePolicyError):
        fs.cat_ranges(["ok", "bad"], [0, 0], [1, 1], on_error="return")
    assert all_records(servers) == []


@pytest.mark.parametrize(
    "ref",
    [
        [None],  # whole-file reference
        ["http://x", 1],  # wrong arity
        ["http://x", "x", 1],  # non-int offset
        ["http://x", 0, -1],  # negative size
        ["http://x", True, 1],  # bool offset
        ["", 0, 1],  # empty URL
        ["http://x", float("nan"), 1],  # non-integral offset
    ],
)
def test_malformed_references_rejected(servers, ref):
    real = url(servers, "http", "h")
    if isinstance(ref[0], str) and ref[0].startswith("http://"):
        ref = [real, *ref[1:]]
    elif len(ref) == 1:
        ref = [real]
    fs = make_ref_fs(servers, {"k": ref}, POLICY)
    with pytest.raises(RangePolicyError):
        fs.cat(["k"])
    assert all_records(servers) == []


def test_unsupported_protocol_rejected_preflight(servers, m):
    m.pipe("/o", b"z" * 10)
    refs = {
        "mem": ("memory://o", 0, 5),
        "h": (url(servers, "http", "h"), 0, 5),
    }
    backends = make_backends()
    backends["memory"] = m
    fs = make_ref_fs(servers, refs, POLICY, backends=backends)
    with pytest.raises(RangePolicyError):
        fs.cat(["mem", "h"])
    with pytest.raises(RangePolicyError):
        fs.cat_ranges(["mem"], [0], [1])
    assert all_records(servers) == []


def test_whole_file_rejected_preflight(servers):
    refs = {
        "whole": [url(servers, "http", "h")],
        "h": (url(servers, "http", "h"), 0, 1),
    }
    fs = make_ref_fs(servers, refs, POLICY)
    with pytest.raises(RangePolicyError):
        fs.cat(["whole", "h"])
    with pytest.raises(RangePolicyError):
        fs.cat_ranges(["whole"], [0], [1])
    assert all_records(servers) == []


def test_slice_out_of_bounds_rejected(servers):
    refs = {"h": (url(servers, "http", "h"), 0, 10)}
    fs = make_ref_fs(servers, refs, POLICY)
    with pytest.raises(RangePolicyError):
        fs.cat_ranges(["h"], [0], [11])
    assert all_records(servers) == []


# --------------------------------------------------------------------------
# missing keys keep the original on_error semantics and never block GETs
# --------------------------------------------------------------------------


def test_missing_key_semantics(servers):
    refs = {"h": (url(servers, "http", "h"), 0, 5)}
    fs = make_ref_fs(servers, refs, POLICY)
    with pytest.raises(FileNotFoundError):
        fs.cat(["missing", "h"])
    out = fs.cat(["missing", "h"], on_error="return")
    assert isinstance(out["missing"], FileNotFoundError)
    assert out["h"] == gen(0, 5)
    out = fs.cat(["missing", "h"], on_error="omit")
    assert list(out) == ["h"]
    n_log = len(servers.http.state.log)
    out = fs.cat_ranges(["missing", "h"], [0, 0], [1, 2])
    assert isinstance(out[0], FileNotFoundError)
    assert out[1] == gen(0, 2)
    assert len(servers.http.state.log) > n_log


# --------------------------------------------------------------------------
# routing: independent backends vs one shared instance
# --------------------------------------------------------------------------


def test_shared_instance_routing_and_headers(servers):
    # one HTTPFileSystem serving both protocols still keeps per-URL
    # identity and routes by scheme; the single instance's session and
    # headers are used for every request
    shared = fsspec.filesystem(
        "http",
        skip_instance_cache=True,
        headers={"X-Tag": "shared-tag"},
        ssl=False,
    )
    refs = four_refs(servers)
    fs = ReferenceFileSystem(
        refs, fs={"http": shared, "https": shared}, range_policy=POLICY
    )
    out = fs.cat(["h0", "h2", "s0", "s2"])
    assert out["s2"] == gen(2, 3)
    for r in all_records(servers):
        assert r["tag"] == "shared-tag"
    # URLs were still routed by scheme to the matching port
    assert {r["scheme"] for r in all_records(servers)} == {"http", "https"}


def test_backend_options_not_rewritten(servers):
    backends = make_backends()
    refs = four_refs(servers)
    fs = make_ref_fs(servers, refs, POLICY, backends=backends)
    fs.cat(["h0", "s0"])
    assert backends["http"].kwargs["headers"]["X-Tag"] == "http-tag"
    assert backends["https"].kwargs["headers"]["X-Tag"] == "https-tag"
    # the reference dict and policy are untouched
    assert refs["h0"] == (url(servers, "http", "h"), 0, 1)
    assert fs.range_policy == POLICY


# --------------------------------------------------------------------------
# ordering, duplicates, containment, slices and cross-request splits
# --------------------------------------------------------------------------


def test_reference_order_permutations(servers):
    refs = four_refs(servers)
    keys = list(refs)
    for order in (keys, list(reversed(keys)), ["s2", "h0", "s0", "h2"]):
        servers.http.state.log.clear()
        servers.https.state.log.clear()
        fs = make_ref_fs(servers, refs, POLICY)
        out = fs.cat(order)
        # every requested key is present and holds its own bytes
        assert set(out) == set(order)
        for k in order:
            u, off, size = refs[k]
            assert out[k] == gen(off, off + size)
        assert sum(declared_bytes(r["range"]) for r in all_records(servers)) <= 5
        # cat_ranges preserves the exact input positions instead
        servers.http.state.log.clear()
        servers.https.state.log.clear()
        positioned = fs.cat_ranges(order, [0] * len(order), [1] * len(order))
        assert positioned == [gen(refs[k][1], refs[k][1] + 1) for k in order]
        assert sum(declared_bytes(r["range"]) for r in all_records(servers)) <= 5


def test_duplicate_and_contained_ranges(servers):
    h = url(servers, "http", "h")
    refs = {
        "a": (h, 0, 100),
        "b": (h, 50, 100),  # overlaps a
        "c": (h, 60, 20),  # contained in both
    }
    policy = {"max_ranges": 8, "max_request_bytes": 1000, "extra_bytes": 0}
    fs = make_ref_fs(servers, refs, policy)
    out = fs.cat(["a", "b", "c", "a"])
    assert out["a"] == gen(0, 100)
    assert out["b"] == gen(50, 150)
    assert out["c"] == gen(60, 80)
    need = union_len([(0, 100), (50, 150), (60, 80)])
    assert sum(declared_bytes(r["range"]) for r in servers.http.state.log) == need
    # cat_ranges: duplicated input positions stay aligned, including
    # repeated logical keys split across different physical slices
    out2 = fs.cat_ranges(["a", "a", "b", "c"], [0, 90, 10, 0], [10, 100, 60, 20])
    assert out2 == [gen(0, 10), gen(90, 100), gen(60, 110), gen(60, 80)]
    # one input spanning more than max_request_bytes is reassembled in
    # input order from several physical requests
    split_policy = {
        "max_ranges": 4,
        "max_request_bytes": 4096,
        "extra_bytes": 0,
    }
    big_url = url(servers, "http", "big", size=30000)
    fs2 = make_ref_fs(servers, {"big": (big_url, 100, 9400)}, split_policy)
    out3 = fs2.cat_ranges(["big"], [0], [9400])
    assert out3 == [gen(100, 9500)]
    assert len(servers.http.state.log) >= 3


def test_logical_slice_offsets_and_zero_length(servers):
    h = url(servers, "http", "h")
    s = url(servers, "https", "s")
    refs = {"h": (h, 100, 100), "s": (s, 200, 50), "i": b"inline-data"}
    fs = make_ref_fs(servers, refs, POLICY)
    out = fs.cat_ranges(
        ["h", "s", "i", "h", "s"],
        [10, 0, 2, 50, 50],
        [20, 50, 6, 50, 50],
    )
    assert out == [
        gen(110, 120),
        gen(200, 250),
        b"line",
        b"",
        b"",
    ]
    # zero-length slices and the inline slice produced no GET
    gets = all_records(servers)
    assert all(r["range"] for r in gets)


def test_key_split_across_requests(servers):
    h = url(servers, "http", "h", size=30000)
    # one 9400-byte logical key with max_request_bytes 4096: 3 requests
    refs = {"big": (h, 100, 9400)}
    policy = {"max_ranges": 4, "max_request_bytes": 4096, "extra_bytes": 0}
    fs = make_ref_fs(servers, refs, policy)
    assert fs.cat("big") == gen(100, 9500)
    gets = [r for r in servers.http.state.log if r["range"]]
    assert len(gets) == 3
    assert sum(declared_bytes(r["range"]) for r in gets) == 9400


def test_inline_only_emits_no_requests(servers):
    fs = make_ref_fs(
        servers,
        {"a": b"data", "b": b"base64:aGVsbG8="},
        POLICY,
    )
    assert fs.cat(["a", "b"]) == {"a": b"data", "b": b"hello"}
    assert fs.cat_ranges(["a", "b"], [1, 0], [3, 2]) == [b"at", b"he"]
    assert all_records(servers) == []


# --------------------------------------------------------------------------
# failure isolation, concurrency, cancellation and instance reuse
# --------------------------------------------------------------------------


def test_one_backend_failure_only_fails_its_keys(servers):
    servers.https.state.fail_paths.add("s")
    refs = four_refs(servers)
    fs = make_ref_fs(servers, refs, POLICY)
    out = fs.cat(["h0", "h2", "s0", "s2"], on_error="return")
    assert out["h0"] == gen(0, 1)
    assert out["h2"] == gen(2, 3)
    assert isinstance(out["s0"], ReferenceNotReachable)
    assert isinstance(out["s2"], ReferenceNotReachable)
    # the HTTP backend's request still happened and carried its header
    http_recs = [r for r in all_records(servers) if r["scheme"] == "http"]
    assert http_recs and all(r["tag"] == "http-tag" for r in http_recs)


def test_concurrent_independent_calls_isolate_budgets(servers):
    refs = four_refs(servers)
    fs = make_ref_fs(servers, refs, POLICY)

    def work():
        return fs.cat(["h0", "h2", "s0", "s2"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outs = list(pool.map(lambda _: work(), range(4)))
    for out in outs:
        assert out["h0"] == gen(0, 1)
        assert out["s2"] == gen(2, 3)
    gets = [r for r in all_records(servers) if r["range"]]
    # each of the four calls obeys its own 5-byte cap: 3 GETs each
    assert len(gets) == 12
    # sanity: calls cannot borrow each other's extra byte
    assert sum(declared_bytes(r["range"]) for r in gets) <= 4 * 5


def test_cancel_then_instance_reusable(servers):
    h = url(servers, "http", "h", size=100000)
    refs = {f"k{j}": (h, 400 * j, 50) for j in range(8)}
    policy = {"max_ranges": 1, "max_request_bytes": 100, "extra_bytes": 0}
    http_fs = fsspec.filesystem(
        "http",
        skip_instance_cache=True,
        headers={"X-Tag": "http-tag"},
        batch_size=2,
    )
    fs = ReferenceFileSystem(refs, fs={"http": http_fs}, range_policy=policy)
    servers.http.state.slow_ms = 300
    urls = [v[0] for v in refs.values()]
    starts = [v[1] for v in refs.values()]
    ends = [v[1] + v[2] for v in refs.values()]
    coro = http_fs._cat_ranges(
        urls, starts, ends, range_policy=policy, on_error="return"
    )
    fut = asyncio.run_coroutine_threadsafe(coro, http_fs.loop)
    time.sleep(0.2)
    fut.cancel()
    with pytest.raises(concurrent.futures.CancelledError):
        fut.result(timeout=10)
    wait_for_quiet(servers)
    # same instance, now responsive again
    servers.http.state.slow_ms = 0
    out = fs.cat(["k0", "k1"])
    assert out["k0"] == gen(0, 50)
    assert out["k1"] == gen(400, 450)


def test_direct_async_http_cat_ranges_uses_shared_budget(servers):
    h = url(servers, "http", "h")
    s = url(servers, "https", "s")
    http_fs, https_fs = make_backends().values()
    # the single extra byte is consumed by the first planner bridging
    # the one-byte gap; the second planner must find nothing left
    budget = CoalesceBudget(1)
    policy = {"max_ranges": 1, "max_request_bytes": 3, "extra_bytes": 1}
    out_h = fsspec.asyn.sync(
        http_fs.loop,
        http_fs._cat_ranges,
        [h, h],
        [0, 2],
        [1, 3],
        range_policy=policy,
        _range_budget=budget,
    )
    assert budget.remaining == 0
    out_s = fsspec.asyn.sync(
        https_fs.loop,
        https_fs._cat_ranges,
        [s, s],
        [0, 2],
        [1, 3],
        range_policy=policy,
        _range_budget=budget,
    )
    assert out_h == [gen(0, 1), gen(2, 3)]
    assert out_s == [gen(0, 1), gen(2, 3)]
    gets = [r for r in all_records(servers) if r["range"]]
    # one coalesced GET (3B) + two single-byte GETs on the second backend
    assert len(gets) == 3
    assert sum(declared_bytes(r["range"]) for r in gets) == 5
    assert sum(1 for r in gets if r["scheme"] == "https") == 2
    # distinct sessions/loops each carried their own instance header
    for r in gets:
        assert r["tag"] == f"{r['scheme']}-tag"


def test_direct_http_cat_ranges_sync_entry(servers):
    h = url(servers, "http", "h")
    http_fs = make_backends()["http"]
    budget = CoalesceBudget(0)
    out = http_fs.cat_ranges(
        [h, h], [0, 2], [1, 3], range_policy=POLICY, _range_budget=budget
    )
    assert out == [gen(0, 1), gen(2, 3)]
    assert len(servers.http.state.log) == 2


def test_bad_budget_type_rejected(servers):
    h = url(servers, "http", "h")
    http_fs = make_backends()["http"]
    with pytest.raises(TypeError):
        http_fs.cat_ranges([h], [0], [1], range_policy=POLICY, _range_budget=object())
    assert all_records(servers) == []


def test_unused_port_network_error_isolated(servers):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    dead = f"http://127.0.0.1:{port}/data?tag=dead"
    good = url(servers, "http", "h")
    refs = {"dead": (dead, 0, 5), "good": (good, 0, 5)}
    fs = make_ref_fs(servers, refs, POLICY)
    out = fs.cat(["dead", "good"], on_error="return")
    assert isinstance(out["dead"], ReferenceNotReachable)
    assert isinstance(out["dead"].__cause__, OSError)
    assert out["good"] == gen(0, 5)
