# Tests for nested savepoints on SimpleCacheFileSystem (fsspec 2025.3.0).
#
# Every test exercises both supported target backends:
#   * a real LocalFileSystem against a pytest temporary directory, and
#   * an isolated in-process MemoryFileSystem (a subclass with its own store,
#     so it never collides with the global ``memory://`` singleton).
#
# The "independent observer" is a second filesystem instance talking directly
# to the same target without sharing the simplecache cache directory, exactly
# like a process that did not open the cache.
from __future__ import annotations

import os
import shutil

import pytest

import fsspec
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem


class IsolatedMemoryFileSystem(MemoryFileSystem):
    """MemoryFileSystem with a private store, usable as an fsspec target."""

    protocol = "mem-isolated"
    store = {}
    pseudo_dirs = [""]


class FailingLocalFileSystem(LocalFileSystem):
    """Local FS whose put_file blows up on demand, to test partial publish."""

    protocol = "file-fail-publish"
    fail_paths: set[str] = set()

    def put_file(self, lpath, rpath, *args, **kwargs):
        if rpath in type(self).fail_paths:
            raise OSError(f"simulated publish failure for {rpath}")
        return super().put_file(lpath, rpath, *args, **kwargs)


@pytest.fixture(params=["local", "memory"])
def backend(request, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    if request.param == "local":
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        target = LocalFileSystem(auto_mkdir=True)
        root = str(target_dir)
        # A separate, un-cached observer of the same on-disk target.
        observer = LocalFileSystem(auto_mkdir=True)
        fs = fsspec.filesystem(
            "simplecache", fs=target, cache_storage=str(cache_dir)
        )
        yield fs, target, observer, root, str(cache_dir)
    else:
        target = IsolatedMemoryFileSystem()
        target.store.clear()
        target.pseudo_dirs.clear()
        target.pseudo_dirs.append("")
        observer = IsolatedMemoryFileSystem()  # shares the class-level store
        root = "/data"
        fs = fsspec.filesystem(
            "simplecache", fs=target, cache_storage=str(cache_dir)
        )
        yield fs, target, observer, root, str(cache_dir)
    target_store = getattr(target, "store", None)
    if target_store is not None:
        target_store.clear()


# --------------------------------------------------------------------------- #
# savepoint() availability / rejection
# --------------------------------------------------------------------------- #


def test_savepoint_outside_transaction_rejects_before_side_effects(tmp_path):
    fs = fsspec.filesystem(
        "simplecache",
        fs=LocalFileSystem(),
        cache_storage=str(tmp_path / "c"),
    )
    with pytest.raises(RuntimeError):
        fs.transaction.savepoint()
    # No transaction working directory was created just by asking.
    assert list(os.listdir(str(tmp_path / "c"))) == []


def test_unsupported_backend_transaction_refuses_savepoint(tmp_path):
    plain = LocalFileSystem()
    with plain.transaction:
        with pytest.raises(RuntimeError):
            plain.transaction.savepoint()
    assert not getattr(plain.transaction, "supports_savepoints", False)


def test_savepoint_transaction_type_is_savepoint_aware(backend):
    fs, _, _, _, _ = backend
    assert fs.transaction_type.supports_savepoints is True


# --------------------------------------------------------------------------- #
# basic nested success / failure
# --------------------------------------------------------------------------- #


def test_nested_savepoint_success_and_rollback(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    target.pipe_file(p("base"), b"base-bytes")

    with fs.transaction:
        fs.pipe_file(p("a"), b"a1")
        # independent observer never sees in-flight writes
        assert not observer.exists(p("a"))

        # inner savepoint completes normally: work stays in the parent
        with fs.transaction.savepoint():
            fs.pipe_file(p("b"), b"b1")
            assert fs.cat(p("b")) == b"b1"
        assert fs.cat(p("b")) == b"b1"
        assert not observer.exists(p("b"))

        # inner savepoint fails: its view is discarded, original error reraised
        with pytest.raises(ValueError, match="branch-boom"):
            with fs.transaction.savepoint():
                fs.pipe_file(p("c"), b"c1")
                fs.rm_file(p("a"))
                assert not fs.exists(p("a"))
                raise ValueError("branch-boom")

        # parent work survived; branch writes vanished
        assert fs.cat(p("a")) == b"a1"
        assert not fs.exists(p("c"))
        assert fs.exists(p("b"))
        assert observer.cat(p("base")) == b"base-bytes"

    # only the outer success publishes
    assert observer.cat(p("a")) == b"a1"
    assert observer.cat(p("b")) == b"b1"
    assert not observer.exists(p("c"))
    assert observer.cat(p("base")) == b"base-bytes"


def test_nested_savepoint_three_levels(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        fs.pipe_file(p("outer"), b"0")
        with fs.transaction.savepoint():
            fs.pipe_file(p("mid"), b"1")
            # an innermost branch that fails restores the mid-level view
            with pytest.raises(RuntimeError):
                with fs.transaction.savepoint():
                    fs.pipe_file(p("inner"), b"2")
                    fs.rm_file(p("outer"))
                    assert not fs.exists(p("outer"))
                    raise RuntimeError("deepest")
            assert fs.exists(p("outer"))
            assert not fs.exists(p("inner"))
            # a second failing branch, then normal mid-level work continues
            with pytest.raises(RuntimeError):
                with fs.transaction.savepoint():
                    fs.pipe_file(p("gone"), b"x")
                    raise RuntimeError("deep")
            assert fs.exists(p("outer"))
            assert not fs.exists(p("gone"))
            assert fs.cat(p("mid")) == b"1"
        assert fs.cat(p("outer")) == b"0"
        assert fs.cat(p("mid")) == b"1"

    assert observer.cat(p("outer")) == b"0"
    assert observer.cat(p("mid")) == b"1"
    assert not observer.exists(p("gone"))
    assert not observer.exists(p("inner"))


def test_outer_rollback_restores_everything(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    observer_fs = observer
    observer_fs.pipe_file(p("keep"), b"orig")

    with pytest.raises(RuntimeError, match="outer-boom"):
        with fs.transaction:
            fs.pipe_file(p("new"), b"n")
            fs.rm_file(p("keep"))
            with fs.transaction.savepoint():
                fs.pipe_file(p("branch"), b"z")
            assert fs.exists(p("branch"))
            raise RuntimeError("outer-boom")

    # target content is exactly as before the transaction
    assert observer.cat(p("keep")) == b"orig"
    assert not observer.exists(p("new"))
    assert not observer.exists(p("branch"))

    # the cache instance cannot resurrect rolled-back data
    assert not fs.exists(p("new"))
    assert not fs.exists(p("branch"))
    assert fs.cat(p("keep")) == b"orig"

    # staging area is gone; only pre-existing cache material may remain
    leftovers = [
        n for n in os.listdir(cache_dir) if n.startswith("tx-")
    ]
    assert leftovers == []

    # the same fs instance is reusable for a fresh transaction
    with fs.transaction:
        fs.pipe_file(p("again"), b"ok")
    assert observer.cat(p("again")) == b"ok"


def test_empty_savepoint_is_a_noop(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        with fs.transaction.savepoint():
            pass
        with fs.transaction.savepoint():
            pass
        fs.pipe_file(p("a"), b"1")
    assert observer.cat(p("a")) == b"1"


# --------------------------------------------------------------------------- #
# view consistency: cat/open, exists/info/ls, mapper
# --------------------------------------------------------------------------- #


def test_views_are_consistent_across_all_read_apis(backend):
    fs, target, observer, root, _ = backend
    mapper = fs.get_mapper(root, create=True)

    def p(name):
        return f"{root}/{name}"

    target.pipe_file(p("old"), b"old-bytes")
    # prewarm the cache for old
    assert fs.cat(p("old")) == b"old-bytes"

    with fs.transaction:
        fs.pipe_file(p("new"), b"new-bytes")
        fs.pipe_file(p("old"), b"overwritten")  # overwrite warmed key
        fs.rm_file(p("gone-marker")) if False else None
        fs.pipe_file(p("sub/deep/key"), b"deep")
        fs.rm_file(p("old"))  # delete previously warmed key, then recreate
        fs.pipe_file(p("old"), b"reborn")

        # cat / open
        assert fs.cat(p("new")) == b"new-bytes"
        with fs.open(p("old"), "rb") as fh:
            assert fh.read() == b"reborn"
        assert fs.info(p("new"))["size"] == len(b"new-bytes")
        assert fs.info(p("old"))["size"] == len(b"reborn")
        assert fs.exists(p("sub/deep/key"))
        assert fs.isfile(p("sub/deep/key"))
        assert fs.isdir(p("sub"))
        assert fs.isdir(p("sub/deep"))

        names = fs.ls(root, detail=False)
        assert p("new") in names and p("old") in names and p("sub") in names
        # no duplicate directory entries
        assert names.count(p("sub")) == 1
        deep_names = fs.ls(p("sub"), detail=False)
        assert deep_names == [p("sub/deep")]

        # FSMap single + iteration + getitems
        mapper["mk"] = b"mv"
        assert mapper["mk"] == b"mv"
        assert "mk" in mapper
        got = mapper.getitems(["mk", "new"])
        assert got == {"mk": b"mv", "new": b"new-bytes"}
        keys = set(mapper)
        assert {"mk", "new", "old", "sub/deep/key"} <= keys

        # target observer still sees the old world
        assert not observer.exists(p("new"))
        assert observer.cat(p("old")) == b"old-bytes"

    assert observer.cat(p("old")) == b"reborn"
    assert observer.cat(p("new")) == b"new-bytes"
    assert observer.cat(p("sub/deep/key")) == b"deep"


def test_repeated_overwrite_same_path(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        for i in range(5):
            fs.pipe_file(p("hot"), f"v{i}".encode() * 100)
        assert fs.cat(p("hot")) == b"v4" * 100
        assert fs.info(p("hot"))["size"] == 2 * 100
        # only one visible entry for the path
        listing = [d for d in fs.ls(root, detail=True) if d["name"] == p("hot")]
        assert len(listing) == 1
    assert observer.cat(p("hot")) == b"v4" * 100


def test_delete_then_recreate(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("k"), b"original")
    fs.cat(p("k"))  # warm

    with fs.transaction:
        fs.rm_file(p("k"))
        assert not fs.exists(p("k"))
        with pytest.raises(FileNotFoundError):
            fs.cat(p("k"))
        fs.pipe_file(p("k"), b"recreated")
        assert fs.cat(p("k")) == b"recreated"
        # an inner delete that fails restores the parent recreation
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                fs.rm_file(p("k"))
                assert not fs.exists(p("k"))
                raise RuntimeError("undo delete")
        assert fs.cat(p("k")) == b"recreated"

    assert observer.cat(p("k")) == b"recreated"


def test_parent_create_child_delete_and_reverse(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        fs.pipe_file(p("parent-made"), b"p")
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                fs.rm_file(p("parent-made"))
                assert not fs.exists(p("parent-made"))
                raise RuntimeError("nope")
        # parent creation restored after child rollback
        assert fs.cat(p("parent-made")) == b"p"

        # child creation that rolls back must not survive
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                fs.pipe_file(p("child-made"), b"c")
                raise RuntimeError("nope")
        assert not fs.exists(p("child-made"))

    assert observer.cat(p("parent-made")) == b"p"
    assert not observer.exists(p("child-made"))


def test_prewarmed_cache_overwrite_and_delete(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("w"), b"warm-original")
    observer.pipe_file(p("d"), b"delete-me")
    assert fs.cat(p("w")) == b"warm-original"
    assert fs.cat(p("d")) == b"delete-me"
    assert len(os.listdir(cache_dir)) >= 2

    with fs.transaction:
        fs.pipe_file(p("w"), b"warm-new")
        fs.rm_file(p("d"))
        assert fs.cat(p("w")) == b"warm-new"
        assert not fs.exists(p("d"))
        with pytest.raises(FileNotFoundError):
            fs.open(p("d"), "rb")

    assert observer.cat(p("w")) == b"warm-new"
    assert not observer.exists(p("d"))
    # cache no longer serves the deleted key
    assert not fs.exists(p("d"))
    with pytest.raises(FileNotFoundError):
        fs.cat(p("d"))


def test_open_write_handle_read_back_through_same_fs(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        with fs.open(p("stream"), "wb") as fh:
            fh.write(b"chunk-1")
            fh.flush()
            # visible while still open
            assert fs.info(p("stream"))["size"] == 7
            fh.write(b"-chunk-2")
        assert fs.cat(p("stream")) == b"chunk-1-chunk-2"
    assert observer.cat(p("stream")) == b"chunk-1-chunk-2"


# --------------------------------------------------------------------------- #
# FSMap batch atomicity
# --------------------------------------------------------------------------- #


def test_fsmap_setitems_halfway_failure_rolls_back(backend):
    fs, target, observer, root, _ = backend
    mapper = fs.get_mapper(root, create=True)

    with fs.transaction:
        mapper["outer"] = b"survive"

        class Bad:
            # maybe_convert only accepts bytes-like / array-like; an arbitrary
            # object reaches pipe_file as-is and fails the raw write.
            pass

        with pytest.raises(TypeError):
            mapper.setitems({"a": b"1", "b": Bad()})
        assert "a" not in mapper
        assert mapper["outer"] == b"survive"

        # plain ints are not bytes-like either and fail at the same boundary
        with pytest.raises(TypeError):
            mapper.setitems({"a": b"1", "c": 123})
        assert "a" not in mapper
        assert mapper["outer"] == b"survive"

    assert observer.exists(f"{root}/outer")
    assert not observer.exists(f"{root}/a")
    assert not observer.exists(f"{root}/c")


def test_fsmap_delitems_missing_key_rolls_back(backend):
    fs, target, observer, root, _ = backend
    mapper = fs.get_mapper(root, create=True)

    with fs.transaction:
        mapper["a"] = b"1"
        mapper["b"] = b"2"
        with pytest.raises(FileNotFoundError):
            mapper.delitems(["a", "missing"])
        # "a" must still be there even though it was processed first
        assert mapper["a"] == b"1"
        assert mapper["b"] == b"2"

        # a successful batch delete is visible inside the same tx
        mapper.delitems(["a", "b"])
        assert "a" not in mapper and "b" not in mapper

    assert not observer.exists(f"{root}/a")
    assert not observer.exists(f"{root}/b")


def test_batch_call_inside_explicit_savepoint(backend):
    fs, target, observer, root, _ = backend
    mapper = fs.get_mapper(root, create=True)

    with fs.transaction:
        with fs.transaction.savepoint():
            mapper.setitems({"x": b"1", "y": b"2"})
            assert mapper.getitems(["x", "y"]) == {"x": b"1", "y": b"2"}
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                mapper.setitems({"z": b"3"})
                raise RuntimeError("drop batch")
        assert "z" not in mapper
        assert mapper["x"] == b"1"

    assert observer.cat(f"{root}/x") == b"1"
    assert not observer.exists(f"{root}/z")


def test_fsmap_behaviour_unchanged_outside_transaction(backend):
    fs, target, observer, root, _ = backend
    mapper = fs.get_mapper(root, create=True)
    mapper.setitems({"a": b"1", "b": b"2"})
    assert observer.cat(f"{root}/a") == b"1"
    mapper.delitems(["a"])
    assert not observer.exists(f"{root}/a")
    with pytest.raises(FileNotFoundError):
        mapper.delitems(["nope"])
    # existing rm semantics outside transactions still surface the error only


# --------------------------------------------------------------------------- #
# abandoned / unclosed write handles
# --------------------------------------------------------------------------- #


def test_rolled_back_handle_cannot_republish_on_late_close(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                fh = fs.open(p("leak"), "wb")
                fh.write(b"secret")
                raise RuntimeError("rollback with open handle")
        # handle stays open across the savepoint boundary; close it late
        fh.close()
        assert not fs.exists(p("leak"))
        with pytest.raises(FileNotFoundError):
            fs.cat(p("leak"))

    assert not observer.exists(p("leak"))


def test_handle_open_across_successful_savepoint_keeps_working(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        with fs.transaction.savepoint():
            fh = fs.open(p("carried"), "wb")
            fh.write(b"first")
        # savepoint merged; continue through the same handle
        fh.write(b"-second")
        fh.close()
        assert fs.cat(p("carried")) == b"first-second"
        # and an outer rollback must disown the carried handle too
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                fh2 = fs.open(p("carried2"), "wb")
                fh2.write(b"x")
                raise RuntimeError("outer-ish")
        fh2.close()
        assert not fs.exists(p("carried2"))

    assert observer.cat(p("carried")) == b"first-second"
    assert not observer.exists(p("carried2"))


# --------------------------------------------------------------------------- #
# publish failure
# --------------------------------------------------------------------------- #


def test_publish_failure_keeps_error_ends_transaction_and_cleans_staging(
    tmp_path,
):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "target"
    cache_dir.mkdir()
    target_dir.mkdir()
    failing = FailingLocalFileSystem()
    FailingLocalFileSystem.fail_paths = {str(target_dir / "boom")}
    fs = fsspec.filesystem(
        "simplecache", fs=failing, cache_storage=str(cache_dir)
    )
    good = str(target_dir / "good")
    boom = str(target_dir / "boom")

    with pytest.raises(OSError, match="simulated publish failure"):
        with fs.transaction:
            fs.pipe_file(good, b"published-first")
            fs.pipe_file(boom, b"never-landed")

    # transaction is finished and cannot be reused as if still active
    assert fs._intrans is False
    assert fs._transaction is None
    # no tx staging directory remains
    assert [n for n in os.listdir(cache_dir) if n.startswith("tx-")] == []
    # already-published file stays, failed one never appears on the target
    assert open(good, "rb").read() == b"published-first"
    assert not os.path.exists(boom)
    # the failed file is not readable through the cache either
    assert not fs.exists(boom)

    FailingLocalFileSystem.fail_paths = set()
    # fs remains usable for a fresh, successful transaction
    with fs.transaction:
        fs.pipe_file(boom, b"retry-ok")
    assert open(boom, "rb").read() == b"retry-ok"


def test_cleanup_failure_is_reported_without_masking_original(backend, monkeypatch):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    real_rmtree = shutil.rmtree

    def flaky_rmtree(path, *args, **kwargs):
        if str(path).startswith(os.path.join(cache_dir, "tx-")):
            # The final destroy() is called without ignore_errors; the
            # per-layer rollback helper uses ignore_errors and must keep
            # working even when removal is flaky.
            if not kwargs.get("ignore_errors", False):
                raise OSError("cleanup denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("fsspec.savepoint.shutil.rmtree", flaky_rmtree)

    # An inner rollback must still succeed (its rmtree ignores errors)...
    with pytest.raises(OSError, match="cleanup denied"):
        with fs.transaction:
            with pytest.raises(RuntimeError):
                with fs.transaction.savepoint():
                    fs.pipe_file(p("a"), b"1")
                    raise RuntimeError("branch")
            assert not fs.exists(p("a"))
            fs.pipe_file(p("b"), b"2")
    # the flaky final teardown still ended the transaction
    assert fs._intrans is False

    monkeypatch.undo()
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)


def test_outer_cleanup_failure_surfaces_and_ends_transaction(backend, monkeypatch):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    real_rmtree = shutil.rmtree

    def flaky_rmtree(path, *args, **kwargs):
        if (
            str(path).startswith(os.path.join(cache_dir, "tx-"))
            and not kwargs.get("ignore_errors", False)
        ):
            raise OSError("cleanup denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("fsspec.savepoint.shutil.rmtree", flaky_rmtree)
    with pytest.raises(OSError, match="cleanup denied"):
        with fs.transaction:
            fs.pipe_file(p("a"), b"1")
    assert fs._intrans is False
    assert fs._transaction is None
    monkeypatch.undo()
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)


# --------------------------------------------------------------------------- #
# ordinary (non-savepoint) usage regressions
# --------------------------------------------------------------------------- #


def test_plain_simplecache_write_without_transaction(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    fs.pipe_file(p("direct"), b"immediate")
    assert observer.cat(p("direct")) == b"immediate"
    with fs.open(p("stream"), "wb") as fh:
        fh.write(b"streamed")
    assert observer.cat(p("stream")) == b"streamed"
    fs.rm_file(p("direct"))
    assert not observer.exists(p("direct"))


def test_legacy_transaction_without_savepoints_still_works(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        fs.pipe_file(p("a"), b"1")
        fs.pipe_file(p("b"), b"22")
        assert not observer.exists(p("a"))
    assert observer.cat(p("a")) == b"1"
    assert observer.cat(p("b")) == b"22"


def test_multirange_read_baseline_preserved(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    payload = b"".join(bytes([i % 256]) for i in range(4096))
    observer.pipe_file(p("range"), payload)
    out = fs.cat_ranges([p("range")], [0], [10])
    assert out[0] == payload[:10]
    out = fs.cat_ranges([p("range"), p("range")], [0, 100], [10, 110])
    assert out[0] == payload[:10]
    assert out[1] == payload[100:110]


def test_cat_multi_path_on_error_semantics(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("a"), b"aaa")
    observer.pipe_file(p("b"), b"bbb")
    out = fs.cat([p("a"), p("b")])
    assert out == {p("a"): b"aaa", p("b"): b"bbb"}


def test_append_and_text_modes_inside_transaction(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("k"), b"base")
    with fs.transaction:
        # append against an existing (not-yet-warmed) key seeds from backend
        with fs.open(p("k"), "ab") as fh:
            fh.write(b"-tail")
        assert fs.cat(p("k")) == b"base-tail"
        # append creating a brand new key
        with fs.open(p("fresh"), "ab") as fh:
            fh.write(b"first")
        # text-mode write
        with fs.open(p("txt"), "w") as fh:
            fh.write("text-value")
        assert fs.cat(p("txt")) == b"text-value"
        # repeated append within nested layers
        with fs.transaction.savepoint():
            with fs.open(p("k"), "ab") as fh:
                fh.write(b"-branch")
            assert fs.cat(p("k")) == b"base-tail-branch"
        with pytest.raises(RuntimeError):
            with fs.transaction.savepoint():
                with fs.open(p("k"), "ab") as fh:
                    fh.write(b"-dropped")
                raise RuntimeError("no")
        assert fs.cat(p("k")) == b"base-tail-branch"
    assert observer.cat(p("k")) == b"base-tail-branch"
    assert observer.cat(p("fresh")) == b"first"
    assert observer.cat(p("txt")) == b"text-value"


def test_savepoint_context_manager_is_single_use(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    with fs.transaction:
        sp = fs.transaction.savepoint()
        with sp:
            fs.pipe_file(p("a"), b"1")
        with pytest.raises(RuntimeError, match="cannot be reused"):
            with sp:
                pass
        # a fresh savepoint still works
        with fs.transaction.savepoint():
            fs.pipe_file(p("b"), b"2")
    assert observer.cat(p("a")) == b"1" and observer.cat(p("b")) == b"2"


def test_recursive_and_list_rm_and_missing_keys(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("d/a"), b"a")
    observer.pipe_file(p("d/sub/b"), b"b")
    observer.pipe_file(p("d/sub/c"), b"c")
    observer.pipe_file(p("single"), b"s")

    with fs.transaction:
        # recursive directory delete: only files are journaled
        fs.rm(p("d"), recursive=True)
        assert not fs.exists(p("d/a"))
        assert not fs.exists(p("d/sub/b"))
        assert not fs.exists(p("d/sub/c"))
        # list delete and single delete
        fs.pipe_file(p("x"), b"1")
        fs.pipe_file(p("y"), b"2")
        fs.rm([p("x"), p("y")])
        assert not fs.exists(p("x")) and not fs.exists(p("y"))
        # missing / double deletes raise, as the backend would
        with pytest.raises(FileNotFoundError):
            fs.rm_file(p("nope"))
        fs.pipe_file(p("twice"), b"t")
        fs.rm_file(p("twice"))
        with pytest.raises(FileNotFoundError):
            fs.rm_file(p("twice"))
        # target untouched until commit
        assert observer.exists(p("d/a")) and observer.exists(p("single"))

    assert not observer.exists(p("d/a"))
    assert not observer.exists(p("d/sub/b"))
    assert observer.cat(p("single")) == b"s"


def test_recursive_rm_rolled_back_inside_savepoint(backend):
    fs, target, observer, root, _ = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("d/a"), b"a")
    observer.pipe_file(p("d/sub/b"), b"b")
    with pytest.raises(RuntimeError):
        with fs.transaction:
            with fs.transaction.savepoint():
                fs.rm(p("d"), recursive=True)
                assert not fs.exists(p("d/a"))
                raise RuntimeError("undo recursive rm")
            assert fs.cat(p("d/a")) == b"a"
            assert fs.cat(p("d/sub/b")) == b"b"
    assert observer.cat(p("d/a")) == b"a"
    assert observer.cat(p("d/sub/b")) == b"b"


def test_cache_miss_read_inside_transaction_does_not_poison_cache(backend):
    fs, target, observer, root, cache_dir = backend

    def p(name):
        return f"{root}/{name}"

    observer.pipe_file(p("r"), b"original")
    # do NOT prewarm: first read happens inside the failed transaction
    with pytest.raises(RuntimeError):
        with fs.transaction:
            assert fs.cat(p("r")) == b"original"  # cache miss, warms in overlay
            fs.pipe_file(p("w"), b"staged")
            raise RuntimeError("roll all back")

    # target unchanged, no staged key, and the rolled-back read materialisation
    # did not leave a transaction work area behind in the cache
    assert observer.cat(p("r")) == b"original"
    assert not fs.exists(p("w"))
    assert [n for n in os.listdir(cache_dir) if n.startswith("tx-")] == []
    # The first post-rollback read fetches fresh bytes from the target.
    assert fs.cat(p("r")) == b"original"
    assert not observer.exists(p("w"))
