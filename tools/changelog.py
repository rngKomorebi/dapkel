"""Extract one release's notes from CHANGELOG.md.

Used by the release workflow to check that the tag being published has a
matching changelog entry, and to fill the GitHub release body from it. Run it
locally before tagging to see exactly what the release notes will say::

    python tools/changelog.py 0.1.0

Exits non-zero (with the available versions listed) when the section is
missing, so a tag can never ship without its changelog entry.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# "## [0.1.0] - 2026-08-01", capturing the version. Also matches a bare
# "## [Unreleased]", which is deliberately not a valid release target.
_HEADING = re.compile(r"^##\s+\[(?P<version>[^\]]+)\]")

DEFAULT_CHANGELOG = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def parse_sections(text: str) -> dict[str, str]:
    """Split a Keep a Changelog file into ``{version: body}``.

    Parameters
    ----------
    text : str
        Full contents of the changelog.

    Returns
    -------
    dict[str, str]
        Section body per version heading, in file order, stripped of
        surrounding blank lines. Link-reference definitions at the foot of the
        file are not part of any section.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = heading.group("version")
            sections[current] = []
            continue
        # Link definitions like "[0.1.0]: https://..." close the last section.
        if current is not None and re.match(r"^\[[^\]]+\]:\s+\S+", line):
            current = None
            continue
        if current is not None:
            sections[current].append(line)

    return {v: "\n".join(body).strip() for v, body in sections.items()}


def notes_for(version: str, changelog: pathlib.Path = DEFAULT_CHANGELOG) -> str:
    """Return the changelog body for ``version``.

    Parameters
    ----------
    version : str
        Release version, with or without a leading ``'v'``.
    changelog : pathlib.Path, optional
        Path to the changelog. Defaults to the one at the repository root.

    Returns
    -------
    str
        The section body.

    Raises
    ------
    SystemExit
        If the version has no section, or the section is empty.
    """
    version = version.lstrip("vV")
    sections = parse_sections(changelog.read_text(encoding="utf-8"))

    if version not in sections:
        released = [v for v in sections if v.lower() != "unreleased"]
        hint = ""
        if sections.get("Unreleased", "").strip():
            # By far the likeliest cause: the notes were written but the
            # section was never renamed, so they are unreleasable as they are.
            hint = (
                f"\n\n'## [Unreleased]' has notes waiting. Promote them:\n"
                f"    python tools/release.py {version}\n"
                f"then commit that and re-create the tag on the new commit."
            )
        sys.exit(
            f"CHANGELOG.md has no '## [{version}]' section.\n"
            f"Add one before tagging. Sections present: {released}{hint}"
        )

    body = sections[version]
    if not body:
        sys.exit(f"CHANGELOG.md section '## [{version}]' is empty.")
    return body


def main() -> None:
    """Print the release notes for the requested version."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="release version, e.g. 0.1.0 or v0.1.0")
    parser.add_argument(
        "--changelog",
        type=pathlib.Path,
        default=DEFAULT_CHANGELOG,
        help="path to CHANGELOG.md (default: repository root)",
    )
    args = parser.parse_args()
    # The notes contain em-dashes and the like. Windows consoles default to
    # cp1252, which mangles them (or raises) when the output is redirected
    # into the release body.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(notes_for(args.version, args.changelog) + "\n")


if __name__ == "__main__":
    main()
