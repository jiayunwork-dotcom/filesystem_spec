"""Tests for reference spec version 2: concatenated (segmented) files.

A version-2 ``refs`` value of ``{"segments": [...]}`` presents the ordered
concatenation of inline byte strings and remote object ranges as one
ordinary file. Ground truth throughout is built from small independent
byte strings, so expected content is a plain bytes concatenation.
"""

import base64
import json
import threading
import tracemalloc
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import fsspec
from fsspec.implementations.reference import (
    ReferenceNotReachable,
    SegmentedReference,
)
from fsspec.spec import AbstractFileSystem

# ---------------------------------------------------------------------------
# test doubles


class RecordingFS(AbstractFileSystem):
    """Serves bytes from an in-memory dict and records every request range."""

    protocol = "rec"

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.requests = []
        self.fail_with = None

    def cat_file(self, path, start=None, end=None, **kwargs):
        if self.fail_with is not None:
            raise self.fail_with
        path = path.split("://", 1)[1] if "://" in path else path
        if path not in self.data:
            raise FileNotFoundError(path)
        self.requests.append((path, start, end))
        return self.data[path][start:end]


def make_fs(refs, backend_data=None, **kwargs):
    """A ReferenceFileSystem over a recording backend (or memory if none)."""
    if backend_data is None:
        m = fsspec.filesystem("memory")
        return fsspec.filesystem("reference", fo=refs, fs=m, **kwargs)
    rec = RecordingFS(backend_data)
    kwargs.setdefault("max_gap", 0)
    return fsspec.filesystem("reference", fo=refs, fs={"rec": rec}, **kwargs), rec


# ---------------------------------------------------------------------------
# format and ground truth


def test_spec_example(m):
    """The worked example: content is the plain concatenation of segments."""
    m.pipe("/a", b"0123456789")
    m.pipe("/b", b"abcdefgh")
    refs = {
        "version": 2,
        "refs": {
            "f": {
                "segments": [
                    ["memory://a", 2, 3],
                    "|",
                    ["memory://b", 1, 2],
                    ["memory://a", 0, 2],
                ]
            }
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    assert fs.cat("f") == b"234|bc01"
    assert fs.cat_file("f", start=2, end=7) == b"4|bc0"


@pytest.fixture()
def truth_setup(m):
    """Small independent source objects and a segmented file over them."""
    a = b"0123456789"
    b = b"abcdefgh"
    m.pipe("/a", a)
    m.pipe("/b", b)
    segments = [
        ["memory://a", 2, 3],  # 234
        "XY",  # XY
        ["memory://b", 0, 4],  # abcd
        ["memory://a", 9, 1],  # 9
        "base64:" + base64.b64encode(b"\x00\xff").decode(),  # b"\x00\xff"
    ]
    truth = a[2:5] + b"XY" + b[0:4] + a[9:10] + b"\x00\xff"
    fs = fsspec.filesystem(
        "reference", fo={"version": 2, "refs": {"f": {"segments": segments}}}, fs=m
    )
    return fs, truth


def test_ground_truth_full(truth_setup):
    fs, truth = truth_setup
    assert fs.cat("f") == truth
    assert fs.cat_file("f") == truth
    assert fs.size("f") == len(truth)
    with fs.open("f", "rb") as f:
        assert f.read() == truth


@pytest.mark.parametrize(
    "start,end",
    [
        (None, None),
        (0, None),
        (None, 5),
        (0, 5),
        (2, 7),
        (3, 4),  # inside one segment
        (4, 6),  # straddling inline/remote boundary
        (5, 6),  # inside inline segment
        (9, 12),  # straddling remote/remote boundary
        (0, 100),  # end beyond EOF
        (100, None),  # start beyond EOF
        (100, 200),  # both beyond EOF
        (-3, None),  # negative start
        (None, -3),  # negative end
        (-100, None),  # negative start beyond beginning
        (None, -100),  # negative end beyond beginning
        (-7, -2),  # negative window
        (5, 5),  # empty
        (7, 2),  # start > end
        (0, 0),
    ],
)
def test_cat_file_slicing(truth_setup, start, end):
    """cat_file start/end are plain bytes-slice bounds of the logical file."""
    fs, truth = truth_setup
    assert fs.cat_file("f", start=start, end=end) == truth[start:end]


def test_read_split_into_adjacent_intervals(truth_setup):
    """Reading in two adjacent intervals equals one read of the whole span."""
    fs, truth = truth_setup
    for cut in range(len(truth) + 1):
        assert fs.cat_file("f", 0, cut) + fs.cat_file("f", cut, None) == truth


def test_segment_splitting_invariance(m):
    """Splitting a segment in two, or inserting zero-length segments,
    does not change the concatenated bytes."""
    m.pipe("/a", b"0123456789")
    base = {"segments": [["memory://a", 0, 6], "XY"]}
    split = {
        "segments": [
            ["memory://a", 0, 3],
            ["memory://a", 3, 3],
            "",
            ["memory://a", 0, 0],
            "X",
            "Y",
            ["memory://a", 5, 0],
        ]
    }
    fs1 = make_fs({"version": 2, "refs": {"f": base}})
    fs2 = make_fs({"version": 2, "refs": {"f": split}})
    assert fs1.cat("f") == fs2.cat("f") == b"012345XY"


def test_duplicate_reversed_overlapping_sources(m):
    """Repeated, reversed and overlapping ranges are kept exactly as declared:
    no de-duplication, no reordering."""
    m.pipe("/a", b"0123456789")
    refs = {
        "version": 2,
        "refs": {
            "f": {
                "segments": [
                    ["memory://a", 5, 3],  # 567
                    ["memory://a", 0, 3],  # 012 (reversed order)
                    ["memory://a", 5, 3],  # 567 (duplicate)
                    ["memory://a", 2, 6],  # 234567 (overlapping both)
                ]
            }
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    assert fs.cat("f") == b"567" + b"012" + b"567" + b"234567"


def test_nonascii_and_base64_inline(m):
    bin_data = bytes(range(256))
    refs = {
        "version": 2,
        "refs": {
            "uni": {"segments": ["héllo", " ", "世界", "!"]},
            "bin": {"segments": ["base64:" + base64.b64encode(bin_data).decode()]},
            "mix": {
                "segments": [
                    "ab",
                    "base64:" + base64.b64encode(b"\x00\x01").decode(),
                    "cd",
                ]
            },
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    assert fs.cat("uni") == "héllo 世界!".encode()
    assert fs.cat("bin") == bin_data
    assert fs.cat("mix") == b"ab\x00\x01cd"
    # sizes count decoded bytes
    assert fs.size("bin") == 256
    assert fs.size("uni") == len("héllo 世界!".encode())


def test_empty_file_and_zero_length_segments(m):
    m.pipe("/a", b"0123456789")
    fs, rec = make_fs(
        {
            "version": 2,
            "refs": {
                "empty": {"segments": []},
                "zeros": {"segments": [["rec://a", 0, 0], "", ["rec://a", 5, 0]]},
            },
        },
        backend_data={"a": b"0123456789"},
    )
    assert fs.cat("empty") == b""
    assert fs.size("empty") == 0
    assert fs.cat("zeros") == b""
    assert fs.size("zeros") == 0
    # zero-length remote segments never reach the backend
    assert rec.requests == []
    with fs.open("empty", "rb") as f:
        assert f.read() == b""


# ---------------------------------------------------------------------------
# templates and target


def test_templates_and_overrides(m):
    m.pipe("/obj", b"0123456789")
    refs = {
        "version": 2,
        "templates": {"u": "memory://obj"},
        "refs": {
            "f": {"segments": [["{{u}}", 2, 3]]},
            "old": ["{{u}}", 0, 2],
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    assert fs.cat("f") == b"234"
    assert fs.cat("old") == b"01"

    fs = fsspec.filesystem(
        "reference", fo=refs, template_overrides={"u": "memory://obj"}, fs=m
    )
    assert fs.cat("f") == b"234"


def test_null_url_uses_target(m):
    m.pipe("/tgt", b"TARGET")
    refs = {"version": 2, "refs": {"f": {"segments": [[None, 1, 3], [None, 0, 1]]}}}
    fs = fsspec.filesystem("reference", fo=refs, target="memory://tgt", fs=m)
    assert fs.cat("f") == b"ARGT"


def test_null_url_without_target_rejected():
    refs = {"version": 2, "refs": {"f": {"segments": [[None, 0, 1]]}}}
    with pytest.raises(ValueError, match="segment 0"):
        fsspec.filesystem("reference", fo=refs)


# ---------------------------------------------------------------------------
# validation


@pytest.mark.parametrize(
    "spec,match",
    [
        ({"segments": [], "extra": 1}, "'f'"),
        ({"segments": "notalist"}, "'f'"),
        ({"segments": {}}, "'f'"),
        ({"segments": [42]}, "segment 0"),
        ({"segments": [None]}, "segment 0"),
        ({"segments": [b"bytes-not-allowed"]}, "segment 0"),
        ({"segments": [{"segments": []}]}, "segment 0"),  # no nesting
        ({"segments": [["memory://a"]]}, "segment 0"),  # no whole-file refs
        ({"segments": [["memory://a", 0]]}, "segment 0"),
        ({"segments": [["memory://a", 0, 1, 2]]}, "segment 0"),
        ({"segments": [["", 0, 1]]}, "segment 0"),  # empty url
        ({"segments": [[123, 0, 1]]}, "segment 0"),  # url not a string
        ({"segments": [["memory://a", -1, 1]]}, "segment 0"),
        ({"segments": [["memory://a", 0, -1]]}, "segment 0"),
        ({"segments": [["memory://a", True, 1]]}, "segment 0"),  # bool not int
        ({"segments": [["memory://a", 0, False]]}, "segment 0"),
        ({"segments": [["memory://a", 0, 1.5]]}, "segment 0"),
        ({"segments": [["memory://a", 0, "1"]]}, "segment 0"),
        ({"segments": ["base64:!!!"]}, "segment 0"),  # invalid base64
        ({"segments": ["base64:abc"]}, "segment 0"),  # bad padding
        ({"segments": [["memory://a", 0, 1], ["memory://a", -2, 1]]}, "segment 1"),
    ],
)
def test_validation_errors(spec, match):
    refs = {"version": 2, "refs": {"f": spec}}
    with pytest.raises(ValueError, match=match):
        fsspec.filesystem("reference", fo=refs)


def test_validation_error_locates_path_and_segment():
    refs = {
        "version": 2,
        "refs": {"some/deep/path": {"segments": ["ok", ["u", 0, -1]]}},
    }
    with pytest.raises(ValueError) as exc:
        fsspec.filesystem("reference", fo=refs)
    assert "some/deep/path" in str(exc.value)
    assert "segment 1" in str(exc.value)


def test_validation_does_not_touch_sources():
    rec = RecordingFS({"a": b"0123456789"})
    refs = {
        "version": 2,
        "refs": {"f": {"segments": [["rec://a", 0, 5], ["rec://missing", 0, 5]]}},
    }
    # construction must not probe the (missing) source
    fs = fsspec.filesystem("reference", fo=refs, fs={"rec": rec})
    assert rec.requests == []
    assert fs.size("f") == 10
    assert rec.requests == []


def test_caller_data_not_mutated(m):
    m.pipe("/a", b"0123456789")
    import copy

    segments = [["memory://a", 2, 3], "|", ["{{u}}", 0, 2]]
    templates = {"u": "memory://a"}
    fo = {
        "version": 2,
        "templates": templates,
        "refs": {"f": {"segments": segments}, "g": ["{{u}}", 0, 2]},
    }
    snapshot = copy.deepcopy(fo)
    fs = fsspec.filesystem(
        "reference", fo=fo, template_overrides={"u": "memory://a"}, fs=m
    )
    assert fs.cat("f") == b"234|01"
    assert fo == snapshot

    bad = {
        "version": 2,
        "templates": templates,
        "refs": {"f": {"segments": [["u", -1, 2]]}},
    }
    bad_snapshot = copy.deepcopy(bad)
    with pytest.raises(ValueError):
        fsspec.filesystem(
            "reference", fo=bad, template_overrides={"u": "memory://a"}, fs=m
        )
    assert bad == bad_snapshot


def test_old_versions_dicts_stay_inline_json(m):
    from fsspec.implementations.reference import json as json_impl

    # version 1: a dict with a "segments" key is still inline JSON
    v1 = {"version": 1, "refs": {"f": {"segments": ["memory://a", 0, 1]}}}
    fs = fsspec.filesystem("reference", fo=v1, fs=m)
    assert fs.cat("f") == json_impl.dumps(v1["refs"]["f"]).encode()

    # version 0: same
    fs = fsspec.filesystem(
        "reference", fo={"f": {"segments": [["memory://a", 0, 1]]}}, fs=m
    )
    assert fs.cat("f") == json_impl.dumps({"segments": [["memory://a", 0, 1]]}).encode()

    # version 2: a dict without "segments" is inline JSON
    fs = fsspec.filesystem(
        "reference", fo={"version": 2, "refs": {"f": {"key": "value"}}}, fs=m
    )
    assert fs.cat("f") == json_impl.dumps({"key": "value"}).encode()


# ---------------------------------------------------------------------------
# metadata


def test_metadata_reports_logical_size_without_remote(m):
    m.pipe("/a", b"0123456789")
    bin_data = b"\x00\xff\x01"
    refs = {
        "version": 2,
        "refs": {
            "dir/seg": {
                "segments": [
                    ["memory://a", 2, 3],
                    "base64:" + base64.b64encode(bin_data).decode(),
                    ["memory://a", 0, 0],
                ]
            },
            "dir/old_range": ["memory://a", 0, 4],
            "dir/old_whole": ["memory://a"],
            "dir/old_inline": b"data",
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    expected = 3 + len(bin_data)
    assert fs.size("dir/seg") == expected
    assert fs.info("dir/seg") == {"name": "dir/seg", "type": "file", "size": expected}
    ls = {e["name"]: e for e in fs.ls("dir", detail=True)}
    assert ls["dir/seg"]["size"] == expected
    assert ls["dir/old_range"]["size"] == 4
    assert ls["dir/old_inline"]["size"] == 4
    assert ls["dir/old_whole"]["size"] is None  # old behavior: unknown until info
    find = fs.find("dir", detail=True)
    assert find["dir/seg"]["size"] == expected
    assert fs.du("dir") == expected + 4 + 4 + 10  # old_whole resolved remotely


# ---------------------------------------------------------------------------
# reads: batch, ranges, handles


def test_cat_batch_mixed_and_repeated(m):
    m.pipe("/a", b"0123456789")
    refs = {
        "version": 2,
        "refs": {
            "seg": {"segments": [["memory://a", 0, 3], "-", ["memory://a", 8, 2]]},
            "old_range": ["memory://a", 0, 2],
            "old_whole": ["memory://a"],
            "old_inline": b"inline",
            "old_json": {"k": 1},
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    out = fs.cat(["seg", "old_range", "old_inline", "seg", "old_whole", "old_json"])
    assert out["seg"] == b"012-89"
    assert out["old_range"] == b"01"
    assert out["old_whole"] == b"0123456789"
    assert out["old_inline"] == b"inline"
    assert out["old_json"] == json.dumps({"k": 1}).encode()
    # single-path form returns the bytes directly
    assert fs.cat("seg") == b"012-89"


def test_cat_ranges_itemwise(m):
    m.pipe("/a", b"0123456789")
    refs = {
        "version": 2,
        "refs": {
            "f": {"segments": [["memory://a", 0, 4], "XY", ["memory://a", 4, 4]]},
            "g": {"segments": [["memory://a", 9, 1]]},
            "bad": {"segments": [["memory://a", 0, 2], ["memory://missing", 0, 2]]},
            "old_range": ["memory://a", 2, 4],
            "old_inline": b"inline",
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    truth = b"0123XY4567"
    # repeated paths, arbitrary order: results correspond item-by-item
    out = fs.cat_ranges(["f", "g", "f", "f"], [0, 0, -2, 3], [4, None, None, 5])
    assert out == [truth[0:4], b"9", truth[-2:], truth[3:5]]
    # old-style references mix freely into the same call
    out = fs.cat_ranges(["f", "old_range", "old_inline"], [1, 1, None], [3, None, None])
    assert out == [truth[1:3], b"2345"[1:], b"inline"]
    # a range hitting the missing source fails as a whole
    with pytest.raises(ReferenceNotReachable):
        fs.cat_ranges(["f", "bad"], [0, 2], [1, 4], on_error="raise")
    out = fs.cat_ranges(["f", "bad"], [0, 2], [1, 4], on_error="return")
    assert out[0] == truth[0:1]
    assert isinstance(out[1], ReferenceNotReachable)
    # a range not intersecting the bad segment still succeeds
    assert fs.cat_ranges(["bad"], [0], [2]) == [b"01"]
    # a path with no reference at all is FileNotFoundError, as for old refs
    with pytest.raises(FileNotFoundError):
        fs.cat_ranges(["missing"], [0], [1], on_error="raise")
    out = fs.cat_ranges(["missing"], [0], [1], on_error="return")
    assert isinstance(out[0], FileNotFoundError)


def test_get_file(m, tmp_path):
    m.pipe("/a", b"0123456789")
    refs = {
        "version": 2,
        "refs": {"f": {"segments": [["memory://a", 2, 3], "|", ["memory://a", 0, 2]]}},
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    dest = str(tmp_path / "out")
    fs.get_file("f", dest)
    with open(dest, "rb") as fo:
        assert fo.read() == b"234|01"


def test_local_backend(tmp_path):
    """Segmented files over the local filesystem, sync and handle reads."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"0123456789")
    b.write_bytes(b"abcdefgh")
    refs = {
        "version": 2,
        "refs": {
            "f": {
                "segments": [
                    [f"file://{a}", 2, 3],
                    "|",
                    [f"file://{b}", 1, 2],
                    [f"file://{a}", 0, 2],
                ]
            }
        },
    }
    fs = fsspec.filesystem("reference", fo=refs)
    truth = b"234|bc01"
    assert fs.cat("f") == truth
    assert fs.cat_file("f", 2, 7) == truth[2:7]
    assert fs.cat_file("f", -3) == truth[-3:]
    with fs.open("f", "rb") as f:
        assert f.read(4) == truth[:4]
        f.seek(-2, 2)
        assert f.read() == truth[-2:]
    # mixed batch over local backend
    out = fs.cat(["f", "f"])
    assert out == {"f": truth}


def test_open_handle_semantics(truth_setup):
    import os as _os

    fs, truth = truth_setup
    f1 = fs.open("f", "rb")
    f2 = fs.open("f", "rb")
    assert f1.read(2) == truth[:2]
    assert f2.read(4) == truth[:4]  # interleaved handles are independent
    assert f1.tell() == 2
    assert f2.tell() == 4

    assert f1.seek(0, _os.SEEK_END) == len(truth)
    assert f1.read() == b""
    assert f1.seek(-3, _os.SEEK_END) == len(truth) - 3
    assert f1.read() == truth[-3:]
    f1.seek(1, _os.SEEK_SET)
    assert f1.read(2) == truth[1:3]
    f1.seek(1, _os.SEEK_CUR)
    assert f1.read(2) == truth[4:6]

    # seek beyond EOF: reads are empty
    f1.seek(len(truth) + 100)
    assert f1.read() == b""
    assert f1.read(10) == b""

    # no seeking to negative positions
    with pytest.raises(ValueError):
        f1.seek(-1)
    with pytest.raises(ValueError):
        f1.seek(-len(truth) - 1, _os.SEEK_END)
    with pytest.raises(ValueError):
        f1.seek(0, 3)  # invalid whence

    # readinto
    f1.seek(0)
    buf = bytearray(4)
    assert f1.readinto(buf) == 4
    assert bytes(buf) == truth[:4]

    # close is idempotent; reads after close fail; other handle unaffected
    f1.close()
    f1.close()
    with pytest.raises(ValueError):
        f1.read(1)
    assert f2.read() == truth[4:]
    f2.close()


def test_open_block_size_and_cache_types():
    data = {"x": b"0123456789" * 100}
    truth = data["x"][0:500] + data["x"][900:1000]
    refs = {
        "version": 2,
        "refs": {"g": {"segments": [["rec://x", 0, 500], ["rec://x", 900, 100]]}},
    }
    fs, rec = make_fs(refs, backend_data=data)

    # cache off: exactly the requested span is fetched
    with fs.open("g", "rb", cache_type="none") as f:
        rec.requests.clear()
        assert f.read(10) == truth[:10]
        assert rec.requests == [("x", 0, 10)]
        f.seek(550)
        rec.requests.clear()
        assert f.read(5) == truth[550:555]
        # logical 550 is 50 bytes into segment 1 -> source offset 950
        assert rec.requests == [("x", 950, 955)]

    # readahead cache may prefetch by (block) size
    with fs.open("g", "rb", block_size=100, cache_type="readahead") as f:
        rec.requests.clear()
        assert f.read(10) == truth[:10]
        assert rec.requests == [("x", 0, 110)]

    # block cache fetches aligned blocks
    with fs.open("g", "rb", block_size=100, cache_type="blockcache") as f:
        rec.requests.clear()
        assert f.read(10) == truth[:10]
        assert rec.requests == [("x", 0, 100)]


def test_random_read_fetches_only_hit_segments():
    data = {"o1": b"0123456789" * 10, "o2": b"abcdefgh" * 10}
    refs = {
        "version": 2,
        "refs": {
            "f": {
                "segments": [
                    ["rec://o1", 0, 5],
                    ["rec://o1", 50, 5],
                    "XX",
                    ["rec://o2", 3, 4],
                ]
            }
        },
    }
    fs, rec = make_fs(refs, backend_data=data)
    truth = data["o1"][0:5] + data["o1"][50:55] + b"XX" + data["o2"][3:7]

    rec.requests.clear()
    assert fs.cat_file("f", 2, 4) == truth[2:4]
    assert rec.requests == [("o1", 2, 4)]  # only the intersecting source span

    rec.requests.clear()
    assert fs.cat_file("f", 3, 12) == truth[3:12]
    # two hit segments, far apart: two requests, no gap bytes
    assert rec.requests == [("o1", 3, 5), ("o1", 50, 55)]

    # untouched segments are not accessed
    rec.requests.clear()
    assert fs.cat_file("f", 12, 16) == truth[12:16]
    assert rec.requests == [("o2", 3, 7)]


def test_merging_of_touching_ranges():
    data = {"o": b"0123456789" * 10}
    refs = {
        "version": 2,
        "refs": {
            "adjacent": {"segments": [["rec://o", 0, 5], ["rec://o", 5, 5]]},
            "overlap": {"segments": [["rec://o", 0, 6], ["rec://o", 4, 6]]},
            "gap": {"segments": [["rec://o", 0, 2], ["rec://o", 5, 2]]},
        },
    }
    fs, rec = make_fs(refs, backend_data=data)

    rec.requests.clear()
    assert fs.cat("adjacent") == b"0123456789"
    assert rec.requests == [("o", 0, 10)]  # adjacent merged into one request

    rec.requests.clear()
    assert fs.cat("overlap") == b"012345" + b"456789"
    assert rec.requests == [("o", 0, 10)]  # overlapping merged

    rec.requests.clear()
    assert fs.cat("gap") == b"01" + b"56"
    assert rec.requests == [("o", 0, 2), ("o", 5, 7)]  # gap not read


def test_max_gap_setting_respected():
    data = {"o": b"0123456789" * 10}
    refs = {
        "version": 2,
        "refs": {"gap": {"segments": [["rec://o", 0, 2], ["rec://o", 5, 2]]}},
    }
    # with a large max_gap the gap bytes are fetched, as for old references
    fs, rec = make_fs(refs, backend_data=data, max_gap=10)
    rec.requests.clear()
    assert fs.cat("gap") == b"01" + b"56"
    assert rec.requests == [("o", 0, 7)]


# ---------------------------------------------------------------------------
# async


@pytest.mark.asyncio
async def test_async_reads(m, tmp_path):
    m.pipe("/a", b"0123456789")
    m.pipe("/b", b"abcdefgh")
    refs = {
        "version": 2,
        "refs": {
            "f": {
                "segments": [
                    ["memory://a", 2, 3],
                    "|",
                    ["memory://b", 1, 2],
                    ["memory://a", 0, 2],
                ]
            },
            "bad": {"segments": [["memory://a", 0, 2], ["memory://missing", 0, 2]]},
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m, asynchronous=True)
    truth = b"234|bc01"

    assert await fs._cat_file("f") == truth
    assert await fs._cat_file("f", start=2, end=7) == truth[2:7]
    assert await fs._cat_file("f", start=-3) == truth[-3:]

    out = await fs._cat(["f", "bad"], on_error="return")
    assert out["f"] == truth
    assert isinstance(out["bad"], ReferenceNotReachable)
    assert out["bad"].reference == "bad"
    assert out["bad"].target == "memory://missing"
    assert isinstance(out["bad"].__cause__, FileNotFoundError)

    out = await fs._cat_ranges(["f", "f", "bad"], [0, 2, 2], [4, 5, 4])
    assert out[0] == truth[0:4]
    assert out[1] == truth[2:5]
    assert isinstance(out[2], ReferenceNotReachable)

    with pytest.raises(ReferenceNotReachable):
        await fs._cat_ranges(["bad"], [2], [4], on_error="raise")

    local = tmp_path / "out"
    await fs._get_file("f", str(local))
    assert local.read_bytes() == truth


# ---------------------------------------------------------------------------
# HTTP backend with recorded Range requests


class RangeHandler(BaseHTTPRequestHandler):
    data = {}
    requests = []
    short_paths = set()

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.__class__.requests.append((self.path, self.headers.get("Range")))
        content = self.data.get(self.path)
        if content is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        ran = self.headers.get("Range")
        if ran:
            _, spec = ran.split("=")
            start_s, end_s = spec.split("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(content) - 1
            out = content[start : end + 1]
            if self.path in self.short_paths:
                out = out[: len(out) // 2]  # truncate: fewer bytes than asked
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)


@pytest.fixture()
def range_server():
    RangeHandler.data = {
        "/obj1": b"0123456789" * 100,
        "/obj2": b"abcdefgh" * 100,
    }
    RangeHandler.requests = []
    RangeHandler.short_paths = set()
    httpd = HTTPServer(("127.0.0.1", 0), RangeHandler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        th.join()
        httpd.socket.close()


def test_http_sync_ranges_and_merging(range_server):
    o1 = b"0123456789" * 100
    o2 = b"abcdefgh" * 100
    refs = {
        "version": 2,
        "refs": {
            "f1": {"segments": [[f"{range_server}/obj1", 10, 10], "||"]},
            "f2": {"segments": [[f"{range_server}/obj1", 20, 10]]},
            "f3": {"segments": [[f"{range_server}/obj2", 500, 10]]},
        },
    }
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=refs, fs=h, max_gap=0)

    RangeHandler.requests.clear()
    out = fs.cat(["f1", "f2", "f3"])
    assert out["f1"] == o1[10:20] + b"||"
    assert out["f2"] == o1[20:30]
    assert out["f3"] == o2[500:510]
    # the two virtual files sharing obj1 merge into one Range request
    obj1_reqs = [r for r in RangeHandler.requests if r[0] == "/obj1"]
    assert obj1_reqs == [("/obj1", "bytes=10-29")]
    assert [r for r in RangeHandler.requests if r[0] == "/obj2"] == [
        ("/obj2", "bytes=500-509")
    ]

    # partial read issues a ranged request for the hit segment only
    RangeHandler.requests.clear()
    assert fs.cat_file("f1", start=0, end=5) == o1[10:15]
    assert RangeHandler.requests == [("/obj1", "bytes=10-14")]

    # file handle over HTTP
    RangeHandler.requests.clear()
    with fs.open("f3", "rb", cache_type="none") as f:
        assert f.read(4) == o2[500:504]
        f.seek(8)
        assert f.read(2) == o2[508:510]
    assert RangeHandler.requests == [
        ("/obj2", "bytes=500-503"),
        ("/obj2", "bytes=508-509"),
    ]


@pytest.mark.asyncio
async def test_http_async(range_server):
    o1 = b"0123456789" * 100
    refs = {
        "version": 2,
        "refs": {
            "f1": {"segments": [[f"{range_server}/obj1", 10, 10], "||"]},
        },
    }
    h = fsspec.filesystem("http", asynchronous=True)
    fs = fsspec.filesystem("reference", fo=refs, fs=h, asynchronous=True)
    assert await fs._cat_file("f1") == o1[10:20] + b"||"
    assert await fs._cat_file("f1", start=1, end=4) == (o1[10:20] + b"||")[1:4]
    out = await fs._cat(["f1"])
    assert out == {"f1": o1[10:20] + b"||"}


def test_http_error_attribution_shared_source(range_server):
    """Two virtual files share a good object; one also reads a missing one.
    The failure is attributed to the right file and the right target URL."""
    o1 = b"0123456789" * 100
    refs = {
        "version": 2,
        "refs": {
            "good": {"segments": [[f"{range_server}/obj1", 0, 5]]},
            "bad": {
                "segments": [
                    [f"{range_server}/obj1", 5, 5],
                    [f"{range_server}/missing", 0, 5],
                ]
            },
        },
    }
    fs = fsspec.filesystem(
        "reference", fo=refs, fs=fsspec.filesystem("http"), max_gap=0
    )
    out = fs.cat(["good", "bad"], on_error="return")
    assert out["good"] == o1[0:5]
    err = out["bad"]
    assert isinstance(err, ReferenceNotReachable)
    assert err.reference == "bad"
    assert err.target == f"{range_server}/missing"
    assert err.__cause__ is not None

    with pytest.raises(ReferenceNotReachable):
        fs.cat(["good", "bad"], on_error="raise")
    out = fs.cat(["good", "bad"], on_error="omit")
    assert out == {"good": o1[0:5]}


def test_http_short_read(range_server):
    RangeHandler.short_paths = {"/obj1"}
    refs = {
        "version": 2,
        "refs": {"f": {"segments": [[f"{range_server}/obj1", 0, 10]]}},
    }
    fs = fsspec.filesystem(
        "reference", fo=refs, fs=fsspec.filesystem("http"), max_gap=0
    )
    with pytest.raises(ReferenceNotReachable) as exc:
        fs.cat("f")
    assert isinstance(exc.value.__cause__, OSError)
    assert "expected 10 bytes, got 5" in str(exc.value.__cause__)


# ---------------------------------------------------------------------------
# failure behavior


def test_missing_source_and_cause(m):
    m.pipe("/a", b"0123456789")
    refs = {
        "version": 2,
        "refs": {"f": {"segments": [["memory://a", 0, 2], ["memory://missing", 0, 3]]}},
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    with pytest.raises(ReferenceNotReachable) as exc:
        fs.cat("f")
    err = exc.value
    assert err.reference == "f"
    assert err.target == "memory://missing"
    assert isinstance(err.__cause__, FileNotFoundError)

    # the part before the missing segment is still readable on its own
    assert fs.cat_file("f", 0, 2) == b"01"
    # a range intersecting the missing segment fails as a whole
    with pytest.raises(ReferenceNotReachable):
        fs.cat_file("f", 1, 3)


def test_raw_io_error_preserved_as_cause():
    backend = RecordingFS({"a": b"0123456789"})
    backend.fail_with = OSError("disk went away")
    refs = {"version": 2, "refs": {"f": {"segments": [["rec://a", 0, 5]]}}}
    fs = fsspec.filesystem("reference", fo=refs, fs={"rec": backend})
    with pytest.raises(ReferenceNotReachable) as exc:
        fs.cat("f")
    assert isinstance(exc.value.__cause__, OSError)
    assert "disk went away" in str(exc.value.__cause__)


def test_short_read_cause():
    class ShortFS(AbstractFileSystem):
        protocol = "short"

        def cat_file(self, path, start=None, end=None, **kwargs):
            return b"ab"

    refs = {"version": 2, "refs": {"f": {"segments": [["short://o", 0, 5]]}}}
    fs = fsspec.filesystem("reference", fo=refs, fs={"short": ShortFS()})
    with pytest.raises(ReferenceNotReachable) as exc:
        fs.cat("f")
    assert isinstance(exc.value.__cause__, OSError)
    assert "expected 5 bytes, got 2" in str(exc.value.__cause__)


def test_batch_error_policies(m):
    m.pipe("/a", b"0123456789")
    refs = {
        "version": 2,
        "refs": {
            "good": {"segments": [["memory://a", 0, 3]]},
            "bad": {"segments": [["memory://a", 0, 2], ["memory://missing", 0, 2]]},
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)

    with pytest.raises(ReferenceNotReachable):
        fs.cat(["good", "bad"], on_error="raise")

    out = fs.cat(["good", "bad"], on_error="return")
    assert out["good"] == b"012"
    assert isinstance(out["bad"], ReferenceNotReachable)

    out = fs.cat(["good", "bad"], on_error="omit")
    assert out == {"good": b"012"}

    # never half-concatenated bytes for a failed file
    out = fs.cat(["bad"], on_error="return")
    assert isinstance(out["bad"], ReferenceNotReachable)


def test_failed_read_does_not_advance_position():
    backend = RecordingFS({"a": b"0123456789"})
    refs = {"version": 2, "refs": {"f": {"segments": [["rec://a", 0, 10]]}}}
    fs = fsspec.filesystem("reference", fo=refs, fs={"rec": backend})
    with fs.open("f", "rb", cache_type="none") as f:
        assert f.read(3) == b"012"
        backend.fail_with = OSError("transient")
        with pytest.raises(ReferenceNotReachable):
            f.read(2)
        assert f.tell() == 3  # position preserved
        backend.fail_with = None
        assert f.read(2) == b"34"  # previously read bytes not re-fetched wrongly


def test_close_isolation_between_handles():
    backend = RecordingFS({"a": b"0123456789"})
    refs = {"version": 2, "refs": {"f": {"segments": [["rec://a", 0, 10]]}}}
    fs = fsspec.filesystem("reference", fo=refs, fs={"rec": backend})
    f1 = fs.open("f", "rb")
    f2 = fs.open("f", "rb")
    f1.read(5)
    f1.close()  # early close
    assert f2.read(4) == b"0123"  # unaffected
    # the borrowed backend filesystem is not closed by the handle
    assert not backend.closed if hasattr(backend, "closed") else True
    assert fs.cat("f") == b"0123456789"  # fs still fully usable
    f2.close()


# ---------------------------------------------------------------------------
# save / reload


def test_save_json_roundtrip(m):
    m.pipe("/a", b"0123456789")
    bin_data = bytes(range(64))
    refs = {
        "version": 2,
        "templates": {"u": "memory://a"},
        "refs": {
            "seg": {
                "segments": [
                    ["{{u}}", 2, 3],
                    "héllo",
                    "base64:" + base64.b64encode(bin_data).decode(),
                    ["memory://a", 0, 0],
                    ["memory://a", 9, 1],
                ]
            },
            "old": ["memory://a", 0, 2],
            "inl": b"data",
        },
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)
    fs.save_json("memory://refs.json")

    saved = json.loads(m.cat("refs.json"))
    assert saved["version"] == 2
    # template-expanded target is what gets saved
    assert saved["refs"]["seg"]["segments"][0] == ["memory://a", 2, 3]

    fs2 = fsspec.filesystem("reference", fo="memory://refs.json", fs=m)
    assert fs2.cat("seg") == fs.cat("seg")
    assert fs2.size("seg") == fs.size("seg")
    assert fs2.references["seg"].segments == fs.references["seg"].segments
    assert fs2.cat("old") == b"01"
    assert fs2.cat("inl") == b"data"


def test_save_json_pure_old_stays_version1(m):
    m.pipe("/a", b"0123456789")
    fs = fsspec.filesystem(
        "reference",
        fo={"version": 1, "refs": {"a": b"data", "b": ["memory://a", 0, 2]}},
        fs=m,
    )
    fs.save_json("memory://old.json")
    assert json.loads(m.cat("old.json"))["version"] == 1


def test_json_file_source_v2(m, tmp_path):
    m.pipe("/a", b"0123456789")
    spec = {
        "version": 2,
        "refs": {"f": {"segments": [["memory://a", 1, 4], "Z"]}},
    }
    path = str(tmp_path / "refs.json")
    with open(path, "w") as fo:
        json.dump(spec, fo)
    fs = fsspec.filesystem("reference", fo=path, fs=m)
    assert fs.cat("f") == b"1234Z"
    assert isinstance(fs.references["f"], SegmentedReference)


# ---------------------------------------------------------------------------
# scale: many segments, huge logical file


class GeneratingFS(AbstractFileSystem):
    """On-demand deterministic byte source; records requests, stores nothing."""

    protocol = "gen"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    @staticmethod
    def byte_at(pos):
        return pos % 251

    def cat_file(self, path, start=None, end=None, **kwargs):
        self.requests.append((path, start, end))
        return bytes(self.byte_at(p) for p in range(start, end))


class CountingSegmentedReference(SegmentedReference):
    """Counts how many segment iterations piece lookups perform."""

    def __init__(self, segments):
        super().__init__(segments)
        self.iterations = 0

    def pieces(self, start, end):
        for piece in super().pieces(start, end):
            self.iterations += 1
            yield piece


@pytest.mark.parametrize("n_segments", [10_000, 100_000])
def test_huge_logical_file(n_segments):
    seg_len = 12_884_901_888  # 12 GiB per segment
    segments = [["gen://big", i * seg_len, seg_len] for i in range(n_segments)]
    gen = GeneratingFS()
    fs = fsspec.filesystem(
        "reference",
        fo={"version": 2, "refs": {"huge": {"segments": segments}}},
        fs={"gen": gen},
        max_gap=0,
    )
    ref = CountingSegmentedReference(fs.references["huge"].segments)
    fs.references["huge"] = ref
    size = ref.size
    assert size >= 2**40  # at least 1 TiB logical

    def truth(pos, n):
        out = b""
        while n > 0:
            take = min(n, (pos // seg_len + 1) * seg_len - pos)
            out += bytes(GeneratingFS.byte_at(p) for p in range(pos, pos + take))
            pos += take
            n -= take
        return out

    import random

    rng = random.Random(1234)
    positions = [0, size - 40] + [rng.randrange(0, size - 40) for _ in range(48)]

    tracemalloc.start()
    ref.iterations = 0
    total_fetched = 0
    for pos in positions:
        gen.requests.clear()
        out = fs.cat_file("huge", start=pos, end=pos + 40)
        assert out == truth(pos, 40)
        # each 40-byte window straddles at most one segment boundary
        assert len(gen.requests) <= 2
        assert all(e - s <= 40 for _, s, e in gen.requests)
        total_fetched += sum(e - s for _, s, e in gen.requests)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # no per-query full scan: iterations are proportional to hit segments
    assert ref.iterations <= 3 * len(positions)
    # memory during reads tracks returned bytes, not the logical file size
    assert peak < 10 * 2**20
    # the backend only ever served the requested windows
    assert total_fetched <= 40 * len(positions)
    print(
        f"\n{n_segments} segments, {size / 2**40:.1f} TiB logical: "
        f"{len(positions)} random 40B reads, {ref.iterations} segment visits, "
        f"{total_fetched} bytes fetched, peak read memory {peak / 2**10:.0f} KiB"
    )


def test_huge_batch_merging_shared_source():
    """A batch of windows over one shared object merges into few requests."""
    gen = GeneratingFS()
    fs = fsspec.filesystem(
        "reference",
        fo={
            "version": 2,
            "refs": {
                # adjacent 5-byte windows: 0-5, 5-10, ... touching ranges
                f"win{i}": {"segments": [["gen://big", i * 5, 5]]}
                for i in range(10)
            },
        },
        fs={"gen": gen},
        max_gap=0,
    )
    gen.requests.clear()
    out = fs.cat([f"win{i}" for i in range(10)])
    for i in range(10):
        expect_pos = i * 5
        assert out[f"win{i}"] == bytes(
            GeneratingFS.byte_at(p) for p in range(expect_pos, expect_pos + 5)
        )
    # ten adjacent 5-byte windows merge into one 50-byte request
    assert gen.requests == [("gen://big", 0, 50)]
