"""The OpenGrad development skills in `.claude/skills/` must describe the repository as it is.

Skills are instructions shipped with the code. They rot in two directions: the repository grows a package or
command no skill explains, or a skill keeps citing a path that was renamed away. Both fail here, so a change
that alters a workflow has to update its skill in the same commit (see `opengrad-skills-maintenance`).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"

#: Console scripts whose subcommands every skill set must cover, with the file that defines them.
SUBCOMMAND_SOURCES = {
    "opengrad-data": "src/opengrad/data/cli.py",
    "opengrad-contamination": "src/opengrad/contamination/cli.py",
    "opengrad-benchmark": "src/opengrad/benchmarks/cli.py",
    "opengrad-annotate": "src/opengrad/annotation/cli.py",
    "python -m opengrad.optimization": "src/opengrad/optimization/__main__.py",
    "opengrad results": "src/opengrad/results/cli.py",
}

#: Top-level directories a backticked token must start with to be checked as a repository path.
PATH_ROOTS = (
    "src/",
    "scripts/",
    "configs/",
    "docs/",
    "reports/",
    "tests/",
    "registry/",
    "release/",
    "results/",
    "runs/",
    "integrations/",
    "hf/",
    "manifests/",
    "experiments/",
    ".github/",
)
PLACEHOLDER_MARKS = ("<", "*", "{", "…", "...")


def _skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS.iterdir() if path.is_dir())


def _skill_files() -> list[Path]:
    return sorted(path for path in SKILLS.rglob("*.md"))


def _corpus() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _skill_files())


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if value:
            fields[key.strip()] = value.strip()
    return fields


def test_skills_exist():
    assert SKILLS.is_dir()
    assert _skill_dirs(), "no skills under .claude/skills"


@pytest.mark.parametrize(
    "skill", _skill_dirs() if SKILLS.is_dir() else [], ids=lambda path: path.name
)
def test_each_skill_is_well_formed(skill: Path):
    skill_md = skill / "SKILL.md"
    assert skill_md.is_file(), f"{skill.name} has no SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    fields = _frontmatter(text)
    assert fields.get("name") == skill.name, "frontmatter name must equal the directory name"
    description = fields.get("description", "")
    assert len(description) >= 120, (
        "the description must say what the skill covers and when to use it"
    )
    assert "This skill should be used" in description
    assert "## Keeping this skill current" in text, "every skill says how it is kept current"


def _backticked_paths(text: str) -> set[str]:
    paths: set[str] = set()
    for token in re.findall(r"`([^`\n]+)`", text):
        candidate = token.strip().split()[0].rstrip(".,;:)")
        if not candidate.startswith(PATH_ROOTS):
            continue
        if any(mark in candidate for mark in PLACEHOLDER_MARKS):
            continue
        paths.add(candidate.split("::")[0])
    return paths


def test_skills_cite_only_paths_that_exist():
    cited = {
        (path, reference)
        for path in _skill_files()
        for reference in _backticked_paths(path.read_text(encoding="utf-8"))
    }
    # A check that examined nothing must not pass: the skills cite far more paths than this.
    assert len(cited) >= 100, f"only {len(cited)} cited paths were examined; the extraction broke"
    missing = {
        f"{path.relative_to(ROOT)}: {reference}"
        for path, reference in cited
        if not (ROOT / reference).exists()
    }
    assert not missing, "skills cite paths that do not exist:\n" + "\n".join(sorted(missing))


def test_skills_cite_only_python_modules_that_exist():
    missing = set()
    for module in re.findall(r"python -m (opengrad(?:\.\w+)+)", _corpus()):
        base = ROOT / "src" / Path(*module.split("."))
        if not (base.with_suffix(".py").is_file() or (base / "__main__.py").is_file()):
            missing.add(module)
    assert not missing, f"skills cite modules that do not exist: {sorted(missing)}"


def test_every_code_package_has_an_owning_skill():
    corpus = _corpus()
    packages = [
        path.name
        for path in sorted((ROOT / "src" / "opengrad").iterdir())
        if path.is_dir()
        and (path / "__init__.py").is_file()
        and any(file.stat().st_size > 0 for file in path.glob("*.py"))
    ]
    uncovered = [
        name
        for name in packages
        if f"src/opengrad/{name}/" not in corpus and f"`{name}/`" not in corpus
    ]
    assert not uncovered, (
        f"packages without a skill: {uncovered}. Add them to the owning skill "
        "(see .claude/skills/opengrad-skills-maintenance)."
    )


def test_every_console_script_has_an_owning_skill():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    corpus = _corpus()
    uncovered = [
        name for name in scripts if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", corpus)
    ]
    assert not uncovered, f"console scripts without a skill: {uncovered}"


def _subcommands(source: str, parser_variable: str | None = "sub") -> list[str]:
    text = (ROOT / source).read_text(encoding="utf-8")
    pattern = (
        rf"(?<!\w){parser_variable}\.add_parser\(\s*\"([\w-]+)\""
        if parser_variable
        else r"add_parser\(\s*\"([\w-]+)\""
    )
    return re.findall(pattern, text)


def test_every_top_level_opengrad_command_has_an_owning_skill():
    corpus = _corpus()
    commands = _subcommands("src/opengrad/cli.py") + ["results"]
    uncovered = [
        name for name in commands if not re.search(rf"opengrad {re.escape(name)}(?![\w-])", corpus)
    ]
    assert not uncovered, f"`opengrad` subcommands without a skill: {uncovered}"


@pytest.mark.parametrize("command", sorted(SUBCOMMAND_SOURCES))
def test_every_subcommand_of_each_tool_has_an_owning_skill(command: str):
    corpus = _corpus()
    names = _subcommands(SUBCOMMAND_SOURCES[command])
    assert names, (
        f"found no subcommands in {SUBCOMMAND_SOURCES[command]}; the parser pattern changed"
    )
    uncovered = []
    for name in names:
        # Either the full invocation, or the subcommand listed after the tool name in a `a | b | c` list.
        full = re.search(rf"{re.escape(command)} {re.escape(name)}(?![\w-])", corpus)
        listed = re.search(
            rf"(?<![\w-]){re.escape(command)}(?![\w-])[^\n]*(?<![\w-]){re.escape(name)}(?![\w-])",
            corpus,
        )
        if not (full or listed):
            uncovered.append(name)
    assert not uncovered, f"`{command}` subcommands without a skill: {uncovered}"
