"""Guards on the packaging metadata and on what the Tests workflow exercises.

'.github/workflows/tests.yml' runs this suite on every Python version that
'pyproject.toml' claims, on Windows and Ubuntu. These tests keep in step the
pieces that running the code cannot check:

    1. 'requirements.txt' mirrors pyproject's dependencies, and every entry is
       capped below its next major version rather than left open or pinned.

    2. The workflow's Python matrix is exactly the set the classifiers claim,
       starting at requires-python's lower bound, which stays uncapped.

    3. Every module imports on the interpreter at hand - the cheapest way to
       catch syntax a newer Python rejects, a removed stdlib module, or a
       dependency API removed at import time.

    4. Every text-mode open() names its encoding. From Python 3.15, UTF-8 is
       the default on every OS; before that, Windows used the ANSI code page,
       so a bare open() reads the same file differently across the matrix.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
SRC = ROOT / "src" / "dapkel"


def _project() -> dict:
    # tomllib is stdlib from 3.11; the other versions in the matrix cover
    # what the 3.10 jobs skip here.
    tomllib = pytest.importorskip("tomllib")
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)["project"]


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


# ---- 1. the two dependency lists -------------------------------------------


def test_requirements_match_pyproject():
    declared = _project()["dependencies"]
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    required = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    assert sorted(required) == sorted(declared)


def test_every_dependency_is_bounded_on_both_sides():
    """'numpy>=1.24' alone installs the next major release, untested.

    An exact pin is the other failure: one pin cannot serve every Python in
    requires-python, so the whole specifier must be a range.
    """
    for spec in _project()["dependencies"]:
        assert ">=" in spec, f"no lower bound: {spec}"
        assert "<" in spec, f"no cap below the next major: {spec}"
        assert "==" not in spec, f"pinned, not capped: {spec}"


# ---- 2. the CI matrix against the claims -----------------------------------


def _matrix_versions() -> list[str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"python-version:\s*\[([^\]]*)\]", text)
    assert match, "no python-version matrix in tests.yml"
    return re.findall(r'"(\d+\.\d+)"', match.group(1))


def test_matrix_is_exactly_the_classifiers():
    claimed = [
        c.rsplit(" ", 1)[1]
        for c in _project()["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    ]
    assert sorted(_matrix_versions(), key=_version_key) == sorted(
        claimed, key=_version_key
    )


def test_matrix_starts_at_requires_python():
    requires = _project()["requires-python"]
    match = re.fullmatch(r">=\s*(\d+\.\d+)", requires)
    # A cap here makes pip, on a newer Python, fall back to an older release
    # of this package that had no cap - worse than a loud install failure.
    assert match, f"requires-python must be a bare lower bound, got {requires!r}"
    assert min(_matrix_versions(), key=_version_key) == match.group(1)


# ---- 3. every module imports ------------------------------------------------


def _all_modules() -> list[str]:
    import dapkel

    names = ["dapkel"]
    for info in pkgutil.walk_packages(dapkel.__path__, prefix="dapkel."):
        if "._old" not in info.name:  # retired code, not packaged
            names.append(info.name)
    return names


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name):
    importlib.import_module(name)


# ---- 4. text opens name their encoding -------------------------------------


def _source_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.py") if "_old" not in p.parts)


def _is_binary_mode(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "b" in node.value
    )


def _text_opens_without_encoding(path: pathlib.Path) -> list[int]:
    """Line numbers of open()/Path.open() calls in text mode lacking encoding=."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            mode = node.args[1] if len(node.args) > 1 else None
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            mode = node.args[0] if node.args else None
        else:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        mode = keywords.get("mode", mode)
        if _is_binary_mode(mode) or "encoding" in keywords:
            continue
        bad.append(node.lineno)
    return bad


@pytest.mark.parametrize(
    "path", _source_files(), ids=lambda p: p.relative_to(SRC).as_posix()
)
def test_text_opens_name_their_encoding(path):
    assert _text_opens_without_encoding(path) == []
