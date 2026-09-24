"""`opengrad.hashing`: the one copy of the sha256 helpers, pinned by literal digests.

The fixtures include CRLF and non-ASCII text on purpose: a helper that normalised line endings or
used a platform encoding would pass an ASCII-only test and still change every recorded hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from opengrad.hashing import CHUNK_BYTES, sha256_bytes, sha256_file, sha256_text

TEXT = "naïve café\r\nline two\n"


def test_sha256_bytes_of_known_inputs() -> None:
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert (
        sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_text_is_utf8_with_no_newline_normalisation() -> None:
    assert sha256_text(TEXT) == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    assert sha256_text(TEXT) != sha256_text(TEXT.replace("\r\n", "\n"))
    assert sha256_text("abc") == sha256_bytes(b"abc")


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(TEXT.encode("utf-8"))
    assert sha256_file(path) == sha256_text(TEXT)


def test_sha256_file_does_not_depend_on_the_chunk_size(tmp_path: Path) -> None:
    data = bytes(range(256)) * (CHUNK_BYTES // 256 * 2 + 3)  # spans three chunks
    path = tmp_path / "large.bin"
    path.write_bytes(data)
    assert len(data) > 2 * CHUNK_BYTES
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
