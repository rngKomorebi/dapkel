"""Promote the CHANGELOG's ``[Unreleased]`` section to a numbered release.

The release gate in '.github/workflows/publish.yml' refuses to publish a tag
whose version has no ``## [<version>]`` section, and it reads the changelog
**as it was at the tagged commit**. So notes written under ``## [Unreleased]``
can never ship: the section has to be renamed, dated and linked *before* the
tag is created. Doing that by hand is the step that gets forgotten, and the
tag then has to be deleted and recreated.

This does the promotion mechanically::

    python tools/release.py 0.2.0          # rewrite CHANGELOG.md
    python tools/release.py 0.2.0 --dry-run   # print the diff instead

It only edits the changelog - it does not commit, tag or push, because the
release itself is cut on the GitHub web UI. The commands to run next are
printed at the end.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

DEFAULT_CHANGELOG = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"

_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+([abrc.].*)?$")
_HEADING = re.compile(r"^##\s+\[(?P<version>[^\]]+)\]")
_UNRELEASED_LINK = re.compile(
    r"^\[Unreleased\]:\s+(?P<base>\S+)/compare/(?P<prev>\S+)\.\.\.HEAD\s*$",
    re.IGNORECASE,
)


def promote(text: str, version: str, date: str) -> str:
    """Rename the ``[Unreleased]`` section to ``[version] - date``.

    The section body is left exactly where it is; only a new heading is
    inserted above it, so an empty ``[Unreleased]`` remains at the top ready
    for the next cycle. The link-reference definitions at the foot of the file
    are updated to match when they follow the Keep a Changelog ``compare``
    form, and left alone when they do not.

    Parameters
    ----------
    text : str
        Full contents of the changelog.
    version : str
        Release version, with or without a leading ``'v'``.
    date : str
        Release date as ``YYYY-MM-DD``.

    Returns
    -------
    str
        The rewritten changelog.

    Raises
    ------
    SystemExit
        If there is no ``[Unreleased]`` heading, if its body is empty (there
        is nothing to release), or if the version already has a section.
    """
    version = version.lstrip("vV")
    lines = text.splitlines()

    start = next(
        (
            i
            for i, line in enumerate(lines)
            if (m := _HEADING.match(line))
            and m.group("version").lower() == "unreleased"
        ),
        None,
    )
    if start is None:
        sys.exit("CHANGELOG.md has no '## [Unreleased]' section to promote.")

    end = next(
        (i for i in range(start + 1, len(lines)) if _HEADING.match(lines[i])),
        len(lines),
    )
    body = "\n".join(lines[start + 1 : end]).strip()
    if not body:
        sys.exit(
            "CHANGELOG.md's '## [Unreleased]' section is empty - there is "
            "nothing to release. Write the notes there first."
        )

    existing = [
        m.group("version") for line in lines if (m := _HEADING.match(line))
    ]
    if version in existing:
        sys.exit(
            f"CHANGELOG.md already has a '## [{version}]' section. Pick a new "
            f"version; sections present: {existing}"
        )

    out = lines[: start + 1] + ["", f"## [{version}] - {date}"] + lines[start + 1 :]
    return _relink(out, version) + "\n"


def _relink(lines: list[str], version: str) -> str:
    """Point ``[Unreleased]`` at the new tag and add the release's own link."""
    for i, line in enumerate(lines):
        match = _UNRELEASED_LINK.match(line)
        if not match:
            continue
        base, prev = match.group("base"), match.group("prev")
        lines[i] = f"[Unreleased]: {base}/compare/v{version}...HEAD"
        lines.insert(
            i + 1, f"[{version}]: {base}/compare/{prev}...v{version}"
        )
        break
    return "\n".join(lines)


def main() -> None:
    """Rewrite CHANGELOG.md so ``version`` becomes a released section."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="release version, e.g. 0.2.0")
    parser.add_argument(
        "--date",
        help="release date, YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--changelog",
        type=pathlib.Path,
        default=DEFAULT_CHANGELOG,
        help="path to CHANGELOG.md (default: repository root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the new section instead of writing the file",
    )
    args = parser.parse_args()

    version = args.version.lstrip("vV")
    if not _VERSION.fullmatch(version):
        sys.exit(
            f"'{args.version}' is not a MAJOR.MINOR.PATCH version. The publish "
            "workflow matches tag 'vX.Y.Z' against the heading, so anything "
            "else can never be released."
        )

    date = args.date or datetime.date.today().isoformat()
    text = args.changelog.read_text(encoding="utf-8")
    promoted = promote(text, version, date)

    if args.dry_run:
        sys.stdout.write(promoted)
        return

    args.changelog.write_text(promoted, encoding="utf-8")
    print(f"CHANGELOG.md: [Unreleased] -> [{version}] - {date}\n")
    print("Check the notes the release will carry:")
    print(f"    python tools/changelog.py {version}\n")
    print("Then commit, and cut the release from the tag on the web UI:")
    print("    git add CHANGELOG.md")
    print(f'    git commit -m "Release {version}"')
    print("    git push\n")
    print(
        "Tag AFTER that commit is on the branch you release from - a tag-\n"
        "triggered run reads the workflow and changelog as they were at the\n"
        "tagged commit, not as they are on main."
    )


if __name__ == "__main__":
    main()
