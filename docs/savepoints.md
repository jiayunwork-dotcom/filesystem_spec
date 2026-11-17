# Nested savepoints for `SimpleCacheFileSystem` transactions

## What this adds

`fsspec` 2025.3.0 transactions only know two states: *buffered* and
*published*. This change adds **nested savepoints** to
`SimpleCacheFileSystem`, so a chunked write can trial a derived branch and, if
the branch fails, undo only that branch while the surrounding transaction keeps
going.

Supported configuration (single process, single writer):

```python
import fsspec

fs = fsspec.filesystem(
    "simplecache",
    fs=fsspec.filesystem("memory"),   # or LocalFileSystem(...)
    cache_storage="./cache",
)

with fs.transaction:                       # outer transaction
    fs.pipe_file("/dataset/chunk-0", b"...")

    with fs.transaction.savepoint():      # trial branch
        fs.pipe_file("/dataset/derived", compute())
        if not happy():
            raise ValueError("discard this branch")
    # normal exit: the branch's bytes stay staged, nothing is uploaded

# only a normal exit of the outer transaction publishes to the target
```

Rules:

| Event | Effect |
| --- | --- |
| savepoint exits normally | its staged writes merge into the parent layer; **no** target upload |
| savepoint raises | the view returns to what it was on entry; the original exception propagates |
| caller catches and continues | the outer transaction can keep writing |
| outer transaction exits normally | the remaining view is published per file through the existing `put`/`rm_file` boundary |
| outer transaction raises | everything is discarded; the target is byte-identical to before the transaction |
| `savepoint()` outside a transaction | `RuntimeError` **before any side effect** (also for non-simplecache filesystems) |

The same rules apply to `FSMap`: single item `d[k] = v` / `del d[k]`, batch
`setitems` / `delitems`, iteration, `getitems` and `in` all see the current
layer's view. Inside an active transaction a `setitems` or `delitems` call is
one local unit — a value-conversion failure, write failure, or deleting a
missing key restores the view from before the call.

Text I/O uses the explicit form `fs.open(path, "r")` / `fs.open(path, "w")`,
which is fully overlay-aware (binary mode is the primary path, as fsspec
always normalises text modes to `b` internally).

## How it is organised

Each transaction owns an **overlay stack** (`fsspec/savepoint.py`). Every open
savepoint is one layer holding a journal of per-path records:

* **write entry** → a private staging file under
  `cache_storage/tx-<pid>-<id>/layer-<n>/`,
* **tombstone** → the logical path reads as absent while visible.

Reads walk the layers innermost-first and resolve a path to a staging file, the
warm cache, the target backend, or *deleted*. Only per-path metadata (a few
strings/booleans) is held in Python; values always live on disk. On savepoint
success the child journal merges into the parent (staging files are cheaply
`os.rename`d within the cache filesystem); on failure the child directory is
removed and its journal dropped.

Responsibility split:

* `fsspec/transaction.py` — base `Transaction.savepoint()` refuses for
  filesystems that cannot honour the contract (`supports_savepoints = False`).
* `fsspec/savepoint.py` — `SavepointTransaction`, the `OverlayStack`, layers,
  entries, merge/rollback and the per-file final publish.
* `fsspec/implementations/cached.py` — `SimpleCacheFileSystem` routes
  `open(..., "wb")`, `pipe_file`, `rm_file`/`rm`, `cat`, `ls`, `info`, `walk`,
  `find`, `mkdir*` and `LocalTempFile` through the active overlay.
* `fsspec/mapping.py` — `FSMap.setitems` / `delitems` wrap themselves in an
  implicit savepoint when the active transaction supports it; behaviour outside
  a transaction is unchanged.

## Trade-offs

* **Repeated overwrites of one large key.** Each overwrite allocates a fresh
  staging file; the previous one is unlinked immediately when its handle is
  closed. This makes rollback deterministic (a layer never mutates a file owned
  by another layer), at the cost of transient disk space proportional to the
  *currently visible* versions plus one in-flight overwrite — never the whole
  dataset, and never buffered in Python.
* **Directory queries.** `ls`/`info` merge the target listing with the in-memory
  journal in O(number of modified paths); no disk scans are needed.
* **Open write handles at a savepoint boundary are supported.** A handle still
  open when its savepoint exits keeps its (renamed) inode; if that layer later
  rolls back the handle is *abandoned* — closing it can never upload or
  resurrect the discarded bytes. This relies on POSIX unlink semantics (Linux).
* **No crash recovery, one writer only** — consistent with the existing
  transaction scope. Deletes cover files; directory entries are implicit.
* **Publish failure** keeps the original backend error, ends the dead
  transaction, removes unpublished staging content, and leaves already
  published files in place (the documented per-file commit boundary).

## Running the tests

```bash
# functional tests, parametrised over LocalFileSystem and an isolated
# MemoryFileSystem target (57 scenarios, each checking cache + observer views)
python3 -m pytest fsspec/implementations/tests/test_savepoint.py -q

# regressions: existing cached-fs, mapper, multi-range read, spec suites
python3 -m pytest fsspec/implementations/tests/test_cached.py \
                 fsspec/tests/test_mapping.py \
                 fsspec/tests/test_multirange.py \
                 fsspec/tests/test_spec.py -q

# resource probe: 64 MiB / 256 MiB values, multi-layer savepoints,
# records tracemalloc peak (Python allocations), RSS peak and staging disk
python3 -I probe_resources.py
python3 -I probe_resources.py --sizes 64          # single size
```

Probe output (representative):

```
value_MiB backend py_extra_MiB rss_peak_MiB outer_staging_MiB inner_staging_MiB rollback_staging_MiB target_MiB
64        memory  0.013         31.7         128.0             224.0              160.0                160.0
64        local   0.012         193.7        128.0             224.0              160.0                160.0
256       memory  0.012         193.7        512.0             896.0              640.0                640.0
256       local   0.012         673.5        512.0             896.0              640.0                640.0
```

`py_extra_MiB` stays at ~12 KiB at both value sizes — the extra active Python
memory depends on the chunk size and number of modified keys, not on total
modified bytes. Opening an inner layer adds only the bytes that layer writes
(here `1.0×` an overwrite plus `0.5×` a new file); unmodified data is never
copied. Rollback releases the discarded layer's staging files.
