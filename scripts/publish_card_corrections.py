#!/usr/bin/env python
"""Re-publish corrected cards to Hugging Face, and record what was published.

A card-only correction replaces a repository's ``README.md`` and nothing else; the weights, and every
revision that pins them, stay as they are. Until 2026-09-24 such corrections were made by hand with
no record, and five of these six repositories carry such a commit from 2026-09-13. This script makes
the operation repeatable:

* **Dry run (default).** For each card: read the **committed** bytes (never the Windows working tree,
  whose line endings differ), find the repository's revision the publication records name as current,
  and walk the Hub's commit list from the live head back to it. Commits after it were published
  without a record; they are listed, and recorded retroactively on upload, as the 2026-09-14 record
  did for the dataset card. The live card is downloaded and a unified diff printed. A ``-`` line in the
  diff is text the live card has and the committed card lacks: uploading would remove it. Nothing is
  written anywhere.
* ``--upload``. Upload each changed card as one commit whose parent is the live head (so a concurrent
  edit fails instead of being overwritten), download the card again at the new revision and require
  it to equal the committed bytes (G17), then write a publication record in ``reports/releases/``.

Exit codes: 0 PASS, 1 FAIL (a re-fetch mismatch, or a card that differs from its generator), 2 BLOCKED
(the recorded revision is not in the Hub's history, a repository is missing, or the network or
credentials are unavailable). Nothing is reported as published without the re-fetch.

    python scripts/publish_card_corrections.py            # dry run: diffs only
    python scripts/publish_card_corrections.py --upload   # publish, verify, record
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes
from opengrad.registry.provenance import read_evidence_bytes
from opengrad.registry.validators import _canonical_repository, _walk_dicts

ROOT = Path(__file__).resolve().parents[1]

#: Card directory -> (Hub repository, repository type). No committed file maps every card to its
#: repository (only the GGUF record names its card), so the table lives here;
#: `tests/publication/test_publish_cards.py` requires each directory to exist and each repository to
#: have a current revision in the publication records.
CARDS: dict[str, tuple[str, str]] = {
    "qwen35-2b-m0-sft-canonicalv2-final": (
        "arjhinety/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final",
        "model",
    ),
    # An evaluation record, not weights: the checkpoints were lost before upload (INC-0001).
    "qwen35-2b-m0-sft-corpusv1-evaluation": (
        "arjhinety/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV1-evaluation",
        "dataset",
    ),
    "qwen35-2b-m1-dpo-canonicalv2-final-v2": (
        "arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2",
        "model",
    ),
    "qwen35-2b-m1-dpo-canonicalv2-final-v2-gguf": (
        "arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2-GGUF",
        "model",
    ),
    "qwen35-2b-m0-abl-minus-xlam-fixed-compute": (
        "arjhinety/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute",
        "model",
    ),
    "qwen35-2b-m0-abl-minus-xlam-matched-exposure": (
        "arjhinety/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure",
        "model",
    ),
}
GGUF_CARD = "qwen35-2b-m1-dpo-canonicalv2-final-v2-gguf"
COMMIT_MESSAGE = "Card correction: license attribution (OpenGrad reports/ERRATA.md §23)"

PASS, FAIL, BLOCKED = 0, 1, 2


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def card_path(directory: str) -> str:
    return f"release/huggingface/{directory}/README.md"


def current_revisions(root: Path = ROOT) -> dict[str, str]:
    """Each repository's current revision: the one recorded and never named as superseded.

    The same rule `opengrad-validate` enforces (`validate_publication_records`), so a repository
    with an ambiguous history is absent here rather than guessed.
    """
    named: dict[str, set[str]] = {}
    superseded: dict[str, set[str]] = {}
    for path in sorted((root / "reports" / "releases").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for _key, value in _walk_dicts(record):
            repository = value.get("repository") or value.get("hub_repository")
            if not isinstance(repository, str) or not repository:
                continue
            repository = _canonical_repository(repository)
            revision = value.get("hub_revision")
            prior = (
                value.get("superseded_hub_revision")
                or value.get("previous_hub_revision")
                or value.get("corrected_hub_revision")
            )
            if isinstance(revision, str) and revision:
                named.setdefault(repository, set()).add(revision)
            if isinstance(prior, str) and prior:
                superseded.setdefault(repository, set()).add(prior)
    current: dict[str, str] = {}
    for repository, revisions in named.items():
        remaining = revisions - superseded.get(repository, set())
        if len(remaining) == 1:
            current[repository] = next(iter(remaining))
    return current


def committed_card(directory: str, root: Path = ROOT) -> bytes:
    data, reason = read_evidence_bytes(root, card_path(directory))
    if data is None:
        raise FileNotFoundError(reason)
    return data


def gguf_card_matches_generator(root: Path = ROOT) -> bool:
    spec = importlib.util.spec_from_file_location(
        "build_gguf_card", root / "scripts/build_gguf_card.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return bool(module.render().encode("utf-8") == committed_card(GGUF_CARD, root))


def github_commit(root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def fetch_card(api: Any, repository: str, repo_type: str, revision: str) -> bytes:
    path = api.hf_hub_download(
        repo_id=repository, filename="README.md", revision=revision, repo_type=repo_type
    )
    return Path(path).read_bytes()


def unrecorded_commits(
    api: Any, repository: str, repo_type: str, recorded: str
) -> list[dict[str, str]] | None:
    """Commits after `recorded`, oldest first, each with its parent; None if `recorded` is absent.

    Hub history on these repositories is linear, so the commit list, newest first, is the chain.
    """
    commits = api.list_repo_commits(repository, repo_type=repo_type)
    ids = [commit.commit_id for commit in commits]
    if recorded not in ids:
        return None
    newer = commits[: ids.index(recorded)]
    chain = []
    parent = recorded
    for commit in reversed(newer):
        chain.append(
            {
                "hub_revision": commit.commit_id,
                "superseded_hub_revision": parent,
                "created_at": commit.created_at.date().isoformat(),
                "title": commit.title,
            }
        )
        parent = commit.commit_id
    return chain


def plan(api: Any, root: Path = ROOT) -> tuple[int, list[dict[str, Any]]]:
    """Compare every committed card with its repository's live card. Writes nothing."""
    if not gguf_card_matches_generator(root):
        print("FAIL: the committed GGUF card differs from scripts/build_gguf_card.py's output")
        return FAIL, []
    recorded = current_revisions(root)
    items: list[dict[str, Any]] = []
    status = PASS
    for directory, (repository, repo_type) in CARDS.items():
        revision = recorded.get(_canonical_repository(repository))
        if revision is None:
            print(f"BLOCKED: {repository} has no single current revision in reports/releases/")
            status = BLOCKED
            continue
        head = api.repo_info(repository, repo_type=repo_type).sha
        chain = unrecorded_commits(api, repository, repo_type, revision)
        if chain is None:
            print(
                f"BLOCKED: {repository}: recorded revision {revision} is not in the Hub's history"
            )
            status = BLOCKED
            continue
        if (chain[-1]["hub_revision"] if chain else revision) != head:
            print(f"BLOCKED: {repository}: the commit list does not end at the head {head}")
            status = BLOCKED
            continue
        local = committed_card(directory, root)
        live = fetch_card(api, repository, repo_type, head)
        changed = local != live
        items.append(
            {
                "directory": directory,
                "repository": repository,
                "repo_type": repo_type,
                "recorded_revision": revision,
                "head": head,
                "unrecorded": chain,
                "card": local,
                "changed": changed,
            }
        )
        print(
            f"== {repository} ({repo_type}) head {head[:12]}, recorded {revision[:12]}: "
            f"{'CHANGED' if changed else 'unchanged'}"
        )
        for commit in chain:
            print(
                f"   unrecorded: {commit['hub_revision'][:12]} {commit['created_at']} "
                f"{commit['title']}"
            )
        if changed:
            diff = difflib.unified_diff(
                live.decode("utf-8").splitlines(keepends=True),
                local.decode("utf-8").splitlines(keepends=True),
                fromfile=f"hub/{repository}/README.md@{head[:12]}",
                tofile=card_path(directory),
            )
            sys.stdout.writelines(diff)
    changed = sum(item["changed"] for item in items)
    unrecorded = sum(len(item["unrecorded"]) for item in items)
    print(
        f"{len(items)} of {len(CARDS)} cards examined, {changed} changed, "
        f"{unrecorded} unrecorded Hub commits found"
    )
    return status, items


def upload(
    api: Any, items: list[dict[str, Any]], root: Path = ROOT
) -> tuple[int, list[dict[str, Any]]]:
    """Publish each changed card, re-fetch it, and return the record entries."""
    today = _today()
    commit = github_commit(root)
    entries: list[dict[str, Any]] = []
    for item in items:
        if not item["changed"] and not item["unrecorded"]:
            continue
        entry: dict[str, Any] = {
            "repository": item["repository"],
            "repo_type": item["repo_type"],
            "revision_chain_since_previous_record": [
                {
                    "repository": item["repository"],
                    "repo_type": item["repo_type"],
                    "hub_revision": link["hub_revision"],
                    "superseded_hub_revision": link["superseded_hub_revision"],
                    "event_occurred_at": link["created_at"],
                    "entry_recorded_at": today,
                    "recorded_retroactively": True,
                    "what": link["title"],
                    "reconstruction_evidence": {
                        "kind": "HUB_COMMIT_HISTORY",
                        "observed_from": (
                            f"huggingface_hub list_repo_commits({item['repository']!r}, "
                            f"repo_type={item['repo_type']!r})"
                        ),
                        "immutable_identifier": link["hub_revision"],
                        "transcribed_not_rederived": True,
                        "contemporaneous_record_exists": False,
                    },
                }
                for link in item["unrecorded"]
            ],
        }
        if item["changed"]:
            try:
                info = api.upload_file(
                    path_or_fileobj=item["card"],
                    path_in_repo="README.md",
                    repo_id=item["repository"],
                    repo_type=item["repo_type"],
                    commit_message=COMMIT_MESSAGE,
                    parent_commit=item["head"],
                )
                revision = info.oid
                refetched = fetch_card(api, item["repository"], item["repo_type"], revision)
            except Exception as exc:  # noqa: BLE001 -- stop here; what was published is kept
                print(f"FAIL: {item['repository']}: {exc}; recording what was published before it")
                return FAIL, entries
            if refetched != item["card"]:
                print(f"FAIL: {item['repository']}@{revision} does not serve the committed card")
                return FAIL, entries
            entry.update(
                {
                    "hub_revision": revision,
                    "superseded_hub_revision": item["head"],
                    "event_occurred_at": today,
                    "entry_recorded_at": today,
                    "recorded_retroactively": False,
                    "files_replaced": ["README.md"],
                    "card_source": card_path(item["directory"]),
                    "card_sha256": sha256_bytes(item["card"]),
                    "github_commit": commit,
                    "refetched_and_matched": True,
                }
            )
            print(f"published {item['repository']} @ {revision} (re-fetched, bytes match)")
        entries.append(entry)
    return PASS, entries


def write_record(entries: list[dict[str, Any]], root: Path = ROOT) -> Path:
    today = _today()
    record = {
        "schema_version": 1,
        "artifact_kind": "HF_PUBLICATION_RECORD",
        "published": today,
        "correction": (
            "Card-only. Each model card gains a license_link to Qwen/Qwen3.5-2B's Apache-2.0 "
            "license at the pinned revision and a License and attribution section, and the "
            "corpus-v1 evaluation card's license becomes other / composite-per-source "
            "(reports/ERRATA.md §23). No weight or other file was written."
        ),
        "cards": entries,
        "why_this_record_exists": (
            "The weights are immutable; their cards are not. The card corrections of 2026-09-13 "
            "were published without a record, which left every recorded revision of these "
            "repositories stale; they are recorded retroactively in each revision chain."
        ),
        "notes": [
            (
                "Each upload named the live head as its parent commit, so it could not overwrite "
                "a change made after the dry run."
            ),
            (
                "Each card was downloaded again at its new revision and compared byte for byte "
                "with the committed file (card_sha256) before this record was written."
            ),
            (
                "A provenance correction is not a new version of a model: the hub_tag study-001 "
                "and every weight revision cited elsewhere still name the same weights."
            ),
        ],
    }
    # Records are append-only: a second correction on the same day gets its own file.
    stem = root / "reports" / "releases" / f"hf-publication-{today}-card-license-correction"
    path = stem.with_suffix(".json")
    number = 2
    while path.exists():
        path = stem.parent / f"{stem.name}-{number}.json"
        number += 1
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--upload", action="store_true", help="publish, re-fetch and record")
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("BLOCKED: huggingface_hub is not installed")
        return BLOCKED
    api = HfApi()
    try:
        status, items = plan(api)
    except Exception as exc:  # noqa: BLE001 -- the network or credentials: nothing is verified
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        return BLOCKED
    if status != PASS or not args.upload:
        return status
    status, entries = upload(api, items)
    if entries:
        print(f"wrote {write_record(entries).relative_to(ROOT).as_posix()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
