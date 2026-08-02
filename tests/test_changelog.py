"""Guards on the changelog the release workflow reads.

'.github/workflows/publish.yml' refuses to publish a tag whose version has no
CHANGELOG.md section, and fills the GitHub release body from it. These tests
keep that contract checkable locally instead of only at release time.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import changelog  # noqa: E402  (needs the sys.path entry above)

CHANGELOG = ROOT / "CHANGELOG.md"


def test_changelog_parses_into_sections():
    sections = changelog.parse_sections(CHANGELOG.read_text(encoding="utf-8"))
    assert "0.0.1" in sections, f"parsed versions: {list(sections)}"


def test_every_released_section_has_content():
    """An empty section would publish a release with empty notes."""
    sections = changelog.parse_sections(CHANGELOG.read_text(encoding="utf-8"))
    empty = [
        v for v, body in sections.items() if v.lower() != "unreleased" and not body
    ]
    assert not empty, f"released versions with no notes: {empty}"


def test_versions_look_like_versions():
    """The workflow maps tag 'vX.Y.Z' onto the heading, so headings must match."""
    sections = changelog.parse_sections(CHANGELOG.read_text(encoding="utf-8"))
    bad = [
        v
        for v in sections
        if v.lower() != "unreleased"
        and not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+([abrc.].*)?", v)
    ]
    assert not bad, f"headings that no tag can match: {bad}"


def test_link_definitions_are_not_part_of_a_section():
    """The trailing '[0.1.0]: https://...' lines must not leak into the notes."""
    oldest = sorted(
        v
        for v in changelog.parse_sections(CHANGELOG.read_text(encoding="utf-8"))
        if v.lower() != "unreleased"
    )[0]
    assert "https://" not in changelog.notes_for(oldest, CHANGELOG)


def test_leading_v_is_stripped():
    assert changelog.notes_for("v0.0.1", CHANGELOG) == changelog.notes_for(
        "0.0.1", CHANGELOG
    )


def test_missing_version_is_a_hard_failure():
    """The library raises; 'main' is what turns this into a non-zero exit.

    Splitting the two is what lets the same check run in a notebook, where a
    'sys.exit' would surface as a SystemExit traceback instead of an error.
    """
    with pytest.raises(changelog.ChangelogError):
        changelog.notes_for("9.9.9", CHANGELOG)

    with pytest.raises(SystemExit) as excinfo:
        changelog.main(["9.9.9", "--changelog", str(CHANGELOG)])
    assert excinfo.value.code != 0
