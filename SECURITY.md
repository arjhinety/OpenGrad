# Security

## Reporting a vulnerability

Report privately to **arjhine@experimentalmachines.org** (the maintainer contact in the README). GitHub's
private vulnerability reporting is not enabled on this repository, so do not open a public issue for a
vulnerability. Include what is affected, how to reproduce it, and its impact. You can expect an
acknowledgement within 7 days.

In scope: the Python package (`src/opengrad`), the scripts, the local annotation server
(`src/opengrad/annotation/server.py`, `integrations/annotate-ui`), the MCP server
(`integrations/opengrad-mcp`), the CI workflow, and any credential or personal data found in a tracked file,
a release or the git history.

## What not to post

Do not include credentials, private datasets, model access tokens or sensitive hardware logs in issues or
pull requests.

## Known, documented findings

A GitHub-token-format string that came with upstream NVIDIA When2Call data sits in one row of the frozen
P-DET population and in one audit tarball. It is not an OpenGrad credential and was reported upstream; the
bytes are kept because the file is pinned evidence ([`reports/ERRATA.md`](reports/ERRATA.md) §22). CI's
hygiene scan (`scripts/repo/check_publication_hygiene.py`) fails on any credential it does not already list.
