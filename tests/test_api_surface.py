"""Structural guards on the package's public API.

These tests do not check any physics - they check that the *shape* of the
package holds as it grows. Two failure modes are guarded:

    1. A public function gets added (or renamed) without being declared in the
       module's ``__all__``, so the declared surface silently drifts from the
       real one.

    2. The same helper gets copy-pasted into a second analysis module instead
       of being lifted into the shared core. This is how the package ended up
       with four copies of '_frames_in_file' and three copies of the heatmap
       plotting skeleton; the test makes the regression loud.

See 'docs/adding_an_analysis.md' for the contract these enforce.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

FUNCTIONS_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "dapkel" / "functions"
)

# The analysis modules, in stage order. 'unpack' is stage 0 and shared by all
# of them, so it is excluded from the duplicate-name check below.
ANALYSIS_MODULES = [
    "calc_diff",
    "data_quality",
    "dcr_analysis",
    "crosstalk_analysis",
    "hitmap_analysis",
    "tdc_calibration",
    "background_subtraction",
    "delta_t",
]

ALL_MODULES = ["unpack", *ANALYSIS_MODULES]


def _parse(module: str) -> ast.Module:
    return ast.parse((FUNCTIONS_DIR / f"{module}.py").read_text(encoding="utf-8"))


def _declared(tree: ast.Module) -> list[str]:
    """Return the module's ``__all__`` entries."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return [e.value for e in node.value.elts]
    return []


def _defined_public(tree: ast.Module) -> set[str]:
    """Public names *defined in this module* (not imported into it)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", None)
                if name and not name.startswith("_") and name != "__all__":
                    names.add(name)
        elif isinstance(node, ast.AnnAssign):
            # 'NAME: type = value' is a different node from 'NAME = value',
            # and missing it made an annotated public constant invisible to
            # this guard - it could neither be required in __all__ nor found
            # when it was declared there.
            name = getattr(node.target, "id", None)
            if name and not name.startswith("_"):
                names.add(name)
    return names


def _top_level_functions(tree: ast.Module) -> set[str]:
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("module", ALL_MODULES)
def test_all_declares_every_public_name(module: str) -> None:
    """``__all__`` must list exactly the public names the module defines."""
    tree = _parse(module)
    declared = set(_declared(tree))
    defined = _defined_public(tree)

    assert declared, f"{module} has no __all__"
    assert not (defined - declared), (
        f"{module}: public but missing from __all__: {sorted(defined - declared)}. "
        f"Add them, or make them private with a leading underscore."
    )
    assert not (declared - defined), (
        f"{module}: in __all__ but not defined here: {sorted(declared - defined)}. "
        f"Re-exporting another module's name hides where it really lives."
    )


def test_no_duplicated_functions_across_analyses() -> None:
    """No function name may be defined in two analysis modules.

    A name defined twice means the implementation was copied. Lift it into the
    shared core instead and import it from there.
    """
    owners: dict[str, list[str]] = {}
    for module in ANALYSIS_MODULES:
        for name in _top_level_functions(_parse(module)):
            owners.setdefault(name, []).append(module)

    duplicated = {n: m for n, m in owners.items() if len(m) > 1}
    assert not duplicated, (
        "These functions are defined in more than one analysis module - lift "
        "them into the shared core and import instead of copying:\n"
        + "\n".join(f"  {n}: {', '.join(m)}" for n, m in sorted(duplicated.items()))
    )


def test_functions_package_reexports_every_analysis_module() -> None:
    """``dapkel.functions.__all__`` must list every analysis module."""
    from dapkel import functions

    assert set(functions.__all__) == set(ALL_MODULES), (
        f"dapkel.functions.__all__ is {sorted(functions.__all__)}, "
        f"expected {sorted(ALL_MODULES)}"
    )
    for module in functions.__all__:
        assert hasattr(functions, module), f"{module} is declared but not imported"


@pytest.mark.parametrize("module", ALL_MODULES)
def test_no_cross_module_private_imports(module: str) -> None:
    """A module must not import another dapkel module's private names.

    Reaching for someone else's ``_helper`` means the helper is really shared
    API and belongs in the core.
    """
    tree = _parse(module)
    offenders = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("dapkel")
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert not offenders, (
        f"{module} imports private names from other dapkel modules: {offenders}. "
        f"Promote them into the shared core instead."
    )


def test_no_hand_built_artifact_paths() -> None:
    """Artifact locations must come from 'core.store', not string joins.

    The package previously scattered stage-1 data across 'senpop_data/',
    'delta_ts_data/' and 'results/tdc_calibration/', with no rule about what
    belonged where. Everything now routes through 'store.processed_dir' /
    'store.results_dir', so the layout is decided in exactly one place.
    """
    banned = ('"results"', "'results'", "senpop_data", "delta_ts_data")
    offenders: list[str] = []
    for module in ALL_MODULES:
        src = (FUNCTIONS_DIR / f"{module}.py").read_text(encoding="utf-8")
        for lineno, line in enumerate(src.split("\n"), start=1):
            code = line.split("#")[0]
            if "os.path.join" in code and any(b in code for b in banned):
                offenders.append(f"{module}.py:{lineno}: {line.strip()}")

    assert not offenders, (
        "Artifact paths must come from core.store (processed_dir / "
        "results_dir), not hand-built joins:\n  " + "\n  ".join(offenders)
    )
