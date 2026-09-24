"""Hand-written documents that repeat pinned revisions must agree with registry/datasets.yaml.

Two documents predate the generated view (docs/datasets/SOURCE_REGISTRY.md) and still carry their
own revision tables: the 2026-09-04 redistribution audit and the normalization-sources reference.
They are kept, not regenerated, because one is a dated decision record. So instead of rewriting
them, this test fails the moment either disagrees with the registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
DOCUMENTS = ("docs/publishing/source-redistribution-audit.md", "docs/data/normalization-sources.md")
REVISION = re.compile(r"`([0-9a-f]{40})`")


def _upstream_revisions() -> set[str]:
    # Both documents cover the training sources; evaluation-only sources are not theirs.
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    return {
        record["source_revision"]["value"]
        for record in registry["datasets"]
        if record["role"] == "UPSTREAM_SOURCE"
        and record["lifecycle"] == "ACTIVE"
        and record["intended_stages"] != ["evaluation"]
    }


def _table_revisions(document: str) -> set[str]:
    text = (ROOT / document).read_text(encoding="utf-8")
    return {m for line in text.splitlines() if line.startswith("|") for m in REVISION.findall(line)}


@pytest.mark.parametrize("document", DOCUMENTS)
def test_the_document_lists_exactly_the_registered_revisions(document: str) -> None:
    listed = _table_revisions(document)
    assert len(listed) == 6, f"{document}: the revision table changed shape"
    assert listed == _upstream_revisions(), document
