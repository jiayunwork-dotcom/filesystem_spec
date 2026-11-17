"""Resource probe for nested savepoints with large values.

Verifies the two scaling promises of the savepoint design:

1. Extra *Python* memory does not grow linearly with the total byte size of
   the modified values.  Values are streamed to private staging files in
   fixed-size chunks, so tracemalloc only sees the small chunk buffers and
   per-key journal metadata.
2. Opening an additional savepoint layer does not copy unmodified data.
   Reads resolve through the outer layer's staging file; only the bytes a
   layer actually writes land in its own directory.

For each value size (64 MiB and 256 MiB) the probe also records the peak
resident set size (resource.getrusage) and the on-disk footprint of the
transaction staging area.

Run:
    python3 -I probe_resources.py [--sizes 64 256]
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import shutil
import sys
import tempfile
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fsspec  # noqa: E402
from fsspec.implementations.local import LocalFileSystem  # noqa: E402
from fsspec.implementations.memory import MemoryFileSystem  # noqa: E402

MIB = 2**20
CHUNK = 4 * MIB  # streaming write granularity - independent of value size


class ProbeMemoryFS(MemoryFileSystem):
    protocol = "mem-probe"
    store = {}
    pseudo_dirs = [""]


def directory_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def stream_pattern(handle, size, pattern):
    """Write *size* bytes through a tiny repeating buffer, never the whole
    value as one Python object."""
    block = (pattern * ((CHUNK // len(pattern)) + 1))[:CHUNK]
    remaining = size
    while remaining:
        piece = block[: min(CHUNK, remaining)]
        handle.write(piece)
        remaining -= len(piece)


def run_one(size_mib, backend_name, workdir):
    size = size_mib * MIB
    cache_dir = os.path.join(workdir, f"cache-{size_mib}mib-{backend_name}")
    os.makedirs(cache_dir)
    if backend_name == "local":
        target_dir = os.path.join(workdir, f"target-{size_mib}mib")
        os.makedirs(target_dir)
        target = LocalFileSystem()
        root = target_dir
    else:
        target = ProbeMemoryFS()
        target.store.clear()
        target.pseudo_dirs.clear()
        target.pseudo_dirs.append("")
        root = "/bulk"

    fs = fsspec.filesystem("simplecache", fs=target, cache_storage=cache_dir)

    gc.collect()
    tracemalloc.start()
    rusage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    snap_before = tracemalloc.take_snapshot()

    with fs.transaction:
        # Outer layer writes two big values via streaming chunked writes.
        with fs.open(f"{root}/big1", "wb") as fh:
            stream_pattern(fh, size, b"outer-one-")
        with fs.open(f"{root}/big2", "wb") as fh:
            stream_pattern(fh, size, b"outer-two-")
        outer_disk = directory_bytes(cache_dir)

        # A nested savepoint overwrites ONE value and creates another. The
        # unmodified big2 must not be copied for the new layer.
        with fs.transaction.savepoint():
            with fs.open(f"{root}/big1", "wb") as fh:
                stream_pattern(fh, size, b"inner-one-")
            with fs.open(f"{root}/big3", "wb") as fh:
                stream_pattern(fh, size // 2, b"inner-three-")
            inner_disk = directory_bytes(cache_dir)
            # Read unmodified data through the overlay: no layer-1 copy made.
            with fs.open(f"{root}/big2", "rb") as fh:
                head = fh.read(16)
            assert head == (b"outer-two-" * 2)[:16]

        snap_inside = tracemalloc.take_snapshot()
        stats = snap_inside.compare_to(snap_before, "lineno")
        py_delta = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        rusage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tracemalloc.stop()

        # Rollback branch: discard another overwrite, confirm bytes return.
        try:
            with fs.transaction.savepoint():
                with fs.open(f"{root}/big1", "wb") as fh:
                    stream_pattern(fh, size, b"dropped-")
                raise RuntimeError("discard this layer")
        except RuntimeError:
            pass
        with fs.open(f"{root}/big1", "rb") as fh:
            assert fh.read(10) == b"inner-one-"
        rolled_back_disk = directory_bytes(cache_dir)

    # Published size on the target
    if backend_name == "local":
        target_bytes = (
            os.path.getsize(os.path.join(target_dir, "big1"))
            + os.path.getsize(os.path.join(target_dir, "big2"))
            + os.path.getsize(os.path.join(target_dir, "big3"))
        )
    else:
        target_bytes = sum(v.size for v in target.store.values())

    return {
        "value_size_mib": size_mib,
        "backend": backend_name,
        "python_extra_bytes": py_delta,
        "rusage_peak_kib": rusage_peak,
        "rusage_before_kib": rusage_before,
        "outer_staging_bytes": outer_disk,
        "inner_staging_bytes": inner_disk,
        "after_rollback_staging_bytes": rolled_back_disk,
        "published_target_bytes": target_bytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 256])
    ap.add_argument(
        "--workdir", default=None, help="base dir (defaults to a temp dir)"
    )
    args = ap.parse_args()
    own = args.workdir is None
    workdir = args.workdir or tempfile.mkdtemp(prefix="savepoint-probe-")
    os.makedirs(workdir, exist_ok=True)

    rows = []
    try:
        for size_mib in args.sizes:
            for backend in ("memory", "local"):
                rows.append(run_one(size_mib, backend, workdir))
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)

    headers = (
        "value_MiB",
        "backend",
        "py_extra_MiB",
        "rss_peak_MiB",
        "outer_staging_MiB",
        "inner_staging_MiB",
        "rollback_staging_MiB",
        "target_MiB",
    )
    print("\t".join(headers))
    for r in rows:
        print(
            "\t".join(
                str(x)
                for x in (
                    r["value_size_mib"],
                    r["backend"],
                    f"{r['python_extra_bytes'] / MIB:.3f}",
                    f"{r['rusage_peak_kib'] / 1024:.1f}",
                    f"{r['outer_staging_bytes'] / MIB:.1f}",
                    f"{r['inner_staging_bytes'] / MIB:.1f}",
                    f"{r['after_rollback_staging_bytes'] / MIB:.1f}",
                    f"{r['published_target_bytes'] / MIB:.1f}",
                )
            )
        )

    # Contract assertions:
    # * extra Python allocations stay two orders of magnitude below one value
    # * while the inner layer is open, the only extra bytes vs. the outer
    #   layer are the bytes that layer itself wrote (overwrite + new file);
    #   unmodified big2 is never duplicated for the new layer
    # * merging and rollback release staging files that are no longer visible
    for r in rows:
        assert r["python_extra_bytes"] < 16 * MIB, r
        added_inner = r["inner_staging_bytes"] - r["outer_staging_bytes"]
        expected_added = r["value_size_mib"] * MIB + (
            r["value_size_mib"] * MIB // 2
        )
        assert abs(added_inner - expected_added) < MIB, r
        assert (
            r["after_rollback_staging_bytes"] <= r["inner_staging_bytes"]
        ), r
    print("\nOK: extra Python memory stays bounded; nesting does not copy "
          "unmodified data; rollback releases discarded staging files.")


if __name__ == "__main__":
    main()
