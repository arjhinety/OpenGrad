"""Vendored, unmodified IFEval instruction checkers from google-research.

Source:  https://github.com/google-research/google-research/tree/master/instruction_following_eval
License: Apache-2.0 (see LICENSE in this directory)
Paper:   Zhou et al., "Instruction-Following Evaluation for Large Language Models", arXiv:2311.07911

The three modules here are byte-for-byte copies of upstream. Their digests are recorded in
`PROVENANCE.json` and asserted by `tests/evaluation/test_ifeval_scoring.py`, so a local edit to a
checker cannot pass unnoticed. They are vendored rather than pip-installed because upstream ships
no distribution on PyPI, and rather than reimplemented because a reimplemented checker would no
longer be the benchmark's own grader.

The package name must stay `instruction_following_eval`: upstream's internal imports are absolute
(`from instruction_following_eval import instructions_util`), and renaming would require editing
the vendored source.
"""
