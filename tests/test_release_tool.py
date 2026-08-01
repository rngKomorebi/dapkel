"""Guards on the changelog promotion the release process depends on.

A tag-triggered publish reads CHANGELOG.md **as it was at the tagged commit**,
so notes sitting under ``## [Unreleased]`` are unreleasable: the section has to
be renamed, dated and linked before the tag exists. 'tools/release.py' does
that; these tests pin the parts that would silently produce a broken release.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import changelog  # noqa: E402  (needs the sys.path entry above)
import release  # noqa: E402

SAMPLE = """\
# Changelog

## [Unreleased]

### Added

- A new thing.

## [0.1.0] - 2026-08-01

Initial.

[Unreleased]: https://github.com/o/r/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/o/r/releases/tag/v0.1.0
"""


def test_promotion_makes_the_notes_readable_by_the_release_gate(tmp_path):
    """The whole point: 'changelog.py <version>' must then succeed."""
    path = tmp_path / "CHANGELOG.md"
    path.write_text(release.promote(SAMPLE, "0.2.0", "2026-08-02"), "utf-8")

    assert changelog.notes_for("0.2.0", path) == "### Added\n\n- A new thing."


def test_unreleased_survives_empty_for_the_next_cycle():
    out = release.promote(SAMPLE, "0.2.0", "2026-08-02")

    assert "## [Unreleased]\n\n## [0.2.0] - 2026-08-02\n" in out
    sections = changelog.parse_sections(out)
    assert sections["Unreleased"] == ""


def test_older_sections_are_untouched():
    sections = changelog.parse_sections(
        release.promote(SAMPLE, "0.2.0", "2026-08-02")
    )
    assert sections["0.1.0"] == "Initial."


def test_links_are_rewritten():
    out = release.promote(SAMPLE, "0.2.0", "2026-08-02")

    assert "[Unreleased]: https://github.com/o/r/compare/v0.2.0...HEAD" in out
    assert "[0.2.0]: https://github.com/o/r/compare/v0.1.0...v0.2.0" in out
    # The superseded Unreleased link must be gone, not merely duplicated.
    assert "compare/v0.1.0...HEAD" not in out


def test_leading_v_is_accepted():
    assert release.promote(SAMPLE, "v0.2.0", "2026-08-02") == release.promote(
        SAMPLE, "0.2.0", "2026-08-02"
    )


def test_empty_unreleased_is_refused():
    """Promoting nothing would publish a release with empty notes."""
    empty = SAMPLE.replace("### Added\n\n- A new thing.\n", "")
    with pytest.raises(SystemExit, match="nothing to release"):
        release.promote(empty, "0.2.0", "2026-08-02")


def test_reusing_a_version_is_refused():
    with pytest.raises(SystemExit, match="already has"):
        release.promote(SAMPLE, "0.1.0", "2026-08-02")


def test_missing_unreleased_heading_is_refused():
    with pytest.raises(SystemExit, match="no '## \\[Unreleased\\]'"):
        release.promote("# Changelog\n\n## [0.1.0] - 2026-08-01\n\nx.\n",
                        "0.2.0", "2026-08-02")


def test_a_changelog_without_links_still_promotes():
    """The link block is conventional, not required - do not crash on it."""
    bare = "# Changelog\n\n## [Unreleased]\n\n- thing\n"
    out = release.promote(bare, "0.2.0", "2026-08-02")
    assert changelog.notes_for(
        "0.2.0", _write(out)
    ) == "- thing"


def _write(text: str) -> pathlib.Path:
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp()) / "CHANGELOG.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_real_changelog_has_no_unreleased_backlog():
    """Notes under [Unreleased] at release time are the bug this all guards.

    Not a hard failure - work in progress belongs there. It fails only if the
    section is left non-empty *and* the newest released section is missing,
    which is the state that breaks a tag.
    """
    sections = changelog.parse_sections(
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    )
    released = [v for v in sections if v.lower() != "unreleased"]
    assert released, "no released section at all - every tag would fail"
