"""Nested savepoints for the write transaction of ``SimpleCacheFileSystem``.

The ordinary :class:`~fsspec.transaction.Transaction` only distinguishes two
states: "buffered, not yet published" and "published".  A chunked dataset that
is written through several trial branches, however, needs a way to discard the
writes of *one* branch while keeping everything the surrounding levels already
produced.  This module implements that on top of the materialised local cache
directory used by :class:`~fsspec.implementations.cached.SimpleCacheFileSystem`.

Design overview
---------------

Each transaction owns an :class:`OverlayStack`.  The stack contains one
:class:`Layer` per open savepoint (the base layer, index ``0``, belongs to the
transaction itself).  Every mutation inside a savepoint is recorded in that
layer's journal as either

* a *write entry* (``kind == "write"``) pointing at a private staging file on
  local disk, or
* a *tombstone* (``kind == "delete"``) remembering that the logical path must
  read as absent while this layer is visible.

Reads never materialise whole values in Python.  A lookup walks the layers from
the innermost outwards and resolves a logical path either to a staging file, to
the pre-warmed file in the cache directory / target backend, or to
:data:`DELETED`.  Only per-path metadata (a handful of strings and booleans) is
held in memory, so active Python memory grows with the *number* of modified
keys, not with their byte size; large values live on disk for their whole
lifetime.  Unmodified data is never copied when an additional savepoint is
opened - the inner layers simply resolve through the outer ones.

On successful savepoint exit the inner journal is merged into the parent
journal (staging files are cheaply renamed into the parent's directory), on an
exception the layer's staging directory is removed and its journal dropped,
restoring the view the caller had when entering the savepoint.  The original
exception always propagates.

Only :class:`SimpleCacheFileSystem` based transactions opt into this contract
via :attr:`SavepointTransaction.supports_savepoints`.  The base
:class:`~fsspec.transaction.Transaction.savepoint` refuses before any state is
changed for filesystems that cannot honour the contract.
"""

from __future__ import annotations

import logging
import os
import shutil
import weakref
from collections import deque

from fsspec.transaction import Transaction

logger = logging.getLogger("fsspec.savepoint")


class _Deleted:
    """Sentinel resolution result for a path hidden by a tombstone."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<DELETED>"


DELETED = _Deleted()


class Entry:
    """One journal record.

    Attributes
    ----------
    path:
        Logical (target-side) path.
    kind:
        ``"write"`` or ``"delete"``.
    fn:
        For write entries, the absolute path of the local staging file holding
        the current bytes.  ``None`` for tombstones.
    existed:
        Whether the target backend held this path when the entry was created.
        Used at final commit time to decide whether a tombstone has to issue a
        backend delete.
    handle:
        Weak reference to the :class:`~fsspec.implementations.cached.LocalTempFile`
        that currently owns the staging file while it is open, else ``None``.
    """

    __slots__ = ("path", "kind", "fn", "existed", "handle")

    def __init__(self, path, kind, fn=None, existed=False, handle=None):
        self.path = path
        self.kind = kind
        self.fn = fn
        self.existed = existed
        self.handle = handle

    def __repr__(self):
        return (
            f"Entry({self.path!r}, {self.kind!r}, fn={self.fn!r}, "
            f"existed={self.existed})"
        )


class Layer:
    """One nesting level of the overlay (a transaction or a savepoint)."""

    def __init__(self, depth, directory):
        self.depth = depth
        # Per-layer directory so a rollback is a single directory removal and
        # can never touch files owned by a parent layer.
        self.directory = directory
        # Logical path -> Entry; at most one visible entry per path per layer.
        self.entries = {}
        # LocalTempFile instances created while this layer was the innermost.
        self.handles = weakref.WeakSet()


class OverlayStack:
    """The stack of layers making up the current transactional view."""

    def __init__(self, fs, workdir):
        self.fs = fs
        self.workdir = workdir
        self.layers = [Layer(0, os.path.join(workdir, "layer-0"))]
        os.makedirs(self.layers[0].directory, exist_ok=True)
        self._seq = 0

    # ------------------------------------------------------------------ stack
    @property
    def current(self):
        """The innermost, currently writable layer."""
        return self.layers[-1]

    @property
    def depth(self):
        return len(self.layers) - 1

    def push(self):
        """Open a new savepoint layer (no side effects beyond mkdir)."""
        depth = len(self.layers)
        directory = os.path.join(self.workdir, f"layer-{depth}")
        os.makedirs(directory, exist_ok=True)
        layer = Layer(depth, directory)
        self.layers.append(layer)
        return layer

    def pop(self):
        return self.layers.pop()

    # --------------------------------------------------------------- mutation
    def _next_name(self, path):
        self._seq += 1
        # The cache mapper already makes a filesystem-safe token per logical
        # path; the sequence suffix keeps repeated overwrites distinct and
        # avoids clobbering a still-open handle.
        return f"{self.fs._mapper(path)}.{os.getpid()}.{self._seq}.tmp"

    def allocate(self, path, handle=None):
        """Register a new write entry in the current layer.

        Returns the absolute staging path the caller should open.  A previous
        entry for the same path in this layer (repeated overwrite, or a
        delete-then-recreate) is replaced; its staging file is removed unless a
        handle still has it open (the handle itself becomes the owner of an
        abandoned, unreadable file).
        """
        layer = self.current
        old = layer.entries.get(path)
        fn = os.path.join(layer.directory, self._next_name(path))
        existed = self._exists_on_target(path)
        entry = Entry(path, "write", fn=fn, existed=existed, handle=handle)
        layer.entries[path] = entry
        if old is not None and old.fn and os.path.lexists(old.fn):
            old_handle = old.handle() if old.handle else None
            if old_handle is None or old_handle.closed:
                try:
                    os.remove(old.fn)
                except FileNotFoundError:
                    pass
        return fn, entry

    def record_delete(self, path):
        """Record a tombstone for *path* in the current layer."""
        layer = self.current
        old = layer.entries.get(path)
        if old is not None and old.fn and os.path.lexists(old.fn):
            old_handle = old.handle() if old.handle else None
            if old_handle is None or old_handle.closed:
                try:
                    os.remove(old.fn)
                except FileNotFoundError:
                    pass
        entry = Entry(path, "delete", existed=self._exists_on_target(path))
        layer.entries[path] = entry
        return entry

    def _exists_on_target(self, path):
        """Existence on the target backend, ignoring this cache's overlay."""
        return self.fs.fs.exists(path)

    # ---------------------------------------------------------------- lookup
    def resolve(self, path):
        """Resolve *path* through the stack, innermost layer first.

        Returns an :class:`Entry`, :data:`DELETED` or ``None`` when the path is
        only known to the target backend / warm cache.
        """
        for layer in reversed(self.layers):
            entry = layer.entries.get(path)
            if entry is not None:
                return DELETED if entry.kind == "delete" else entry
        return None

    def is_deleted(self, path):
        return self.resolve(path) is DELETED

    def effective_entries(self):
        """Collapse the stack into the visible ``{path: Entry}`` mapping.

        Tombstones win over outer writes; used for ``ls``/``info`` and for the
        final commit.  Cost is O(number of modified paths).
        """
        view = {}
        for layer in self.layers:
            for path, entry in layer.entries.items():
                view[path] = entry
        return view

    # ------------------------------------------------------------ merge/roll
    def merge_child(self, child):
        """Fold a successfully closed savepoint layer into its parent."""
        parent = self.layers[-1]
        # Handles still open at the savepoint boundary keep writing into the
        # renamed inode; make sure a later parent rollback disowns them too.
        for handle in list(child.handles):
            parent.handles.add(handle)
        for path, entry in child.entries.items():
            old = parent.entries.get(path)
            if entry.kind == "write":
                new_fn = os.path.join(
                    parent.directory, os.path.basename(entry.fn)
                )
                if os.path.lexists(new_fn):
                    # Extremely unlikely (sequence counter), but stay correct.
                    new_fn = os.path.join(parent.directory, self._next_name(path))
                os.rename(entry.fn, new_fn)
                entry.fn = new_fn
                parent.entries[path] = entry
                if old is not None and old.fn and old.fn != new_fn:
                    self._safe_remove(old)
            else:  # tombstone
                parent.entries[path] = entry
                if old is not None and old.fn:
                    self._safe_remove(old)
        child.entries.clear()
        shutil.rmtree(child.directory, ignore_errors=True)

    def discard_layer(self, layer):
        """Remove every staging file produced by *layer* and drop its journal."""
        # Handles created by this layer must never be able to publish again.
        for handle in list(layer.handles):
            handle._mark_abandoned()
        shutil.rmtree(layer.directory, ignore_errors=True)
        layer.entries.clear()

    @staticmethod
    def _safe_remove(entry):
        if entry.fn and os.path.lexists(entry.fn):
            handle = entry.handle() if entry.handle else None
            if handle is not None and not handle.closed:
                # Let the open handle keep its private inode; it is no longer
                # referenced by any journal, so closing it cannot republish.
                return
            try:
                os.remove(entry.fn)
            except FileNotFoundError:
                pass

    def size_of(self, path):
        """Current visible size of *path* without reading its bytes."""
        resolved = self.resolve(path)
        if resolved is DELETED or resolved is None:
            return None
        handle = resolved.handle() if resolved.handle else None
        if handle is not None and not handle.closed:
            pos = handle.fh.tell()
            try:
                handle.fh.flush()
                return os.fstat(handle.fh.fileno()).st_size
            except (OSError, ValueError):
                return pos
        try:
            return os.path.getsize(resolved.fn)
        except OSError:
            return None

    def destroy(self):
        """Remove the whole overlay working directory.

        Raises the underlying ``OSError`` if removal fails, after every open
        handle has been disowned, so the caller can report it while the
        transaction itself is already unusable.
        """
        for layer in self.layers:
            for handle in list(layer.handles):
                handle._mark_abandoned()
        shutil.rmtree(self.workdir)


class _SavepointCM:
    def __init__(self, transaction):
        self.tx = transaction
        self.layer = None
        self._used = False

    def __enter__(self):
        if self._used:
            raise RuntimeError(
                "a savepoint() context manager cannot be reused; call "
                "transaction.savepoint() again for a new branch"
            )
        tx = self.tx
        if tx.fs is None or tx._closed or not getattr(tx.fs, "_intrans", False):
            raise RuntimeError(
                "savepoint() can only be used inside an active "
                "`with fs.transaction:` block"
            )
        self._used = True
        self.layer = tx._overlay.push()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Merge the child's view into the parent; nothing is uploaded.
            child = self.tx._overlay.pop()
            self.tx._overlay.merge_child(child)
        else:
            child = self.tx._overlay.pop()
            self.tx._overlay.discard_layer(child)
        # Never swallow the exception.
        return False


class SavepointTransaction(Transaction):
    """Transaction with nested savepoints for ``SimpleCacheFileSystem``.

    The final publish still happens per file through the target backend's
    ``put``/``rm_file`` boundary, exactly like :class:`WriteCachedTransaction`;
    savepoints only rearrange which staged files survive until that point.
    """

    supports_savepoints = True

    def __init__(self, fs, **kwargs):
        super().__init__(fs, **kwargs)
        # The overlay (and its working directory) is created lazily in
        # start(), so merely touching ``fs.transaction`` outside a
        # ``with`` block has no side effects on disk.
        self._workdir = None
        self._overlay = None
        self._closed = False

    # ------------------------------------------------------------- entrypoint
    def savepoint(self):
        """Return a context manager scoping a nested, discardable branch.

        Raises ``RuntimeError`` before any side effect when called outside an
        active transaction.
        """
        if self.fs is None or self._closed or not self.fs._intrans:
            raise RuntimeError(
                "savepoint() can only be used inside an active "
                "`with fs.transaction:` block"
            )
        return _SavepointCM(self)

    def start(self):
        """Create the overlay working area and mark the transaction active."""
        if self._overlay is None:
            workdir = os.path.join(
                self.fs.storage[-1], f"tx-{os.getpid()}-{id(self)}"
            )
            os.makedirs(workdir, exist_ok=True)
            self._workdir = workdir
            self._overlay = OverlayStack(self.fs, workdir)
        self.fs._intrans = True

    # ---------------------------------------------------------------- commit
    def complete(self, commit=True):
        """Publish or discard everything, preserving per-file I/O boundaries."""
        overlay = self._overlay
        fs = self.fs
        original_error = None
        try:
            if overlay is not None and fs is not None and commit:
                self._publish(overlay)
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            cleanup_error = None
            if overlay is not None:
                try:
                    overlay.destroy()
                except OSError as exc:
                    # The transaction is dead either way; remember the cleanup
                    # failure to surface after the state has been reset, but
                    # never mask the error that got us here.
                    cleanup_error = exc
            if fs is not None:
                fs._intrans = False
                fs._transaction = None
            self._closed = True
            self.fs = None
            self.files = deque()
            if original_error is None and cleanup_error is not None:
                raise cleanup_error

    def _publish(self, overlay):
        fs = self.fs
        view = overlay.effective_entries()

        writes = [e for e in view.values() if e.kind == "write"]
        deletes = [e for e in view.values() if e.kind == "delete" and e.existed]

        publish_error = None
        published = set()
        try:
            if writes:
                # Ensure target parents exist; this happens at publish time,
                # so no namespace change leaks to an independent observer any
                # earlier than the file bytes themselves.
                parents = {
                    fs.fs._parent(e.path)
                    for e in writes
                    if fs.fs._parent(e.path).strip("/")
                }
                for parent in sorted(parents, key=lambda p: p.count("/")):
                    fs.fs.makedirs(parent, exist_ok=True)
                # One batched put, but the backend still applies put_file per
                # path: a failure leaves the already-published files in place,
                # matching the documented per-file commit boundary.
                fs.fs.put(
                    [e.fn for e in writes],
                    [e.path for e in writes],
                )
                published.update(e.path for e in writes)
            for entry in deletes:
                fs.fs.rm_file(entry.path)
                published.add(entry.path)
            # All backend I/O succeeded: make the staging files the new warm
            # cache and drop any stale cache copy of a deleted key.
            for entry in writes:
                self._promote_to_cache(entry)
            for entry in deletes:
                fs._evict_cache_file(entry.path)
        except BaseException as exc:
            publish_error = exc
            # The cache view must not keep serving bytes for files that never
            # made it to the backend.
            for entry in writes:
                if entry.path not in published:
                    fs._evict_cache_file(entry.path)
            for entry in deletes:
                if entry.path not in published:
                    fs._evict_cache_file(entry.path)
        if publish_error is not None:
            # Preserve the original backend error; complete() will still tear
            # down the now-useless staging area in its finally block.
            raise publish_error

    def _promote_to_cache(self, entry):
        """Move a published staging file into the ordinary cache directory."""
        fs = self.fs
        target_fn = os.path.join(fs.storage[-1], fs._mapper(entry.path))
        try:
            if os.path.abspath(entry.fn) != os.path.abspath(target_fn):
                if os.path.lexists(target_fn):
                    os.remove(target_fn)
                os.rename(entry.fn, target_fn)
        except OSError:
            # Caching is best-effort; the backend already has the bytes.
            logger.debug("could not promote %s into cache", entry.path, exc_info=True)
