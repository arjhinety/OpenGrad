"""The sha256 helpers OpenGrad's library code and scripts share.

Before 2026-09-24 about forty modules and scripts each defined their own copy of these three
functions. They were byte-for-byte equivalent, but a copy can drift (a different chunk size is
harmless, a forgotten ``encode("utf-8")`` is not), so one copy is kept here and pinned by literal
digests in `tests/test_hashing.py`.

Some helpers are deliberately *not* this module:

* the ones whose source bytes are recorded by hash (the frozen classifiers, the P-DET builders,
  ``normalization_v3.CODE_MODULES``), because editing the file changes the recorded hash;
* the ones that hash something other than raw bytes (LF-normalised source, canonical JSON with a
  module-specific serialiser, selected fields), because their meaning differs;
* standalone scripts and the Modal container scripts, which run without the package installed.

`tests/repo/test_hash_helpers.py` lists every remaining local sha256 helper with its reason and fails
on a new one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Files are read in 1 MiB chunks; the digest does not depend on the chunk size.
CHUNK_BYTES = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hex sha256 of ``text`` encoded as UTF-8, with no newline normalisation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex sha256 of the file's exact bytes, read in chunks so a large file is never held whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
