"""Hugging Face repositories that hold private material and must never be the target of a publish step.

The run-log dataset holds every run's output log. Its redaction is best effort (``opengrad.runrecords``), so the
only thing keeping a missed credential or home folder private is that the repository stays private. The run-record
upload checks that at upload time; every other script that writes to the Hub calls ``refuse_private_only`` first, so
a release, card correction or artifact upload can never write into it, or create it as a public repository.
``tests/publication/test_private_repos.py`` fails when a tracked file makes one of the Hub-write calls it lists
without naming this check. It cannot see an aliased or indirect call, or prove the check runs before the write.
"""

from __future__ import annotations

#: Repositories only ``python -m opengrad.runrecords upload`` may write to (owner's decision, 2026-11-13). The
#: account's former name ``arrochi112`` (``opengrad.registry.validators.NAMESPACE_ALIASES``) is listed too.
PRIVATE_ONLY_REPOS = frozenset({"arjhinety/opengrad-run-logs", "arrochi112/opengrad-run-logs"})


class PrivateRepoError(RuntimeError):
    """A publish step named a repository that must stay private."""


def is_private_only(repo_id: str) -> bool:
    """True for a repository on ``PRIVATE_ONLY_REPOS``. Hub repository names are compared case-insensitively."""
    return repo_id.strip().strip("/").lower() in PRIVATE_ONLY_REPOS


def refuse_private_only(repo_id: str) -> None:
    """Raise ``PrivateRepoError`` when a publish step names a private-only repository."""
    if is_private_only(repo_id):
        raise PrivateRepoError(
            f"{repo_id!r} holds private run logs and must never be a publish target "
            "(opengrad.publication.private_repos)"
        )
