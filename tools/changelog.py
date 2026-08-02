"""Extract one release's notes from CHANGELOG.md.

Used by the release workflow to check that the tag being published has a
matching changelog entry, and to fill the GitHub release body from it. Run it
locally before tagging to see exactly what the release notes will say::

    python tools/changelog.py 0.1.0

Exits non-zero (with the available versions listed) when the section is
missing, so a tag can never ship without its changelog entry.

The checks live in importable functions, so they also run in a Jupyter
interactive window. Run this file there and it defines the functions rather
than firing the CLI - argparse would otherwise parse the *kernel's* argv and
die with ``SystemExit: 2`` - then call them::

    notes_for("0.2.0")          # the release body for a version
    main(["0.2.0"])             # the CLI, with an explicit argv

Failures raise 'ChangelogError' rather than calling ``sys.exit``. Only 'main'
turns that into a non-zero exit, so the workflow's gate is unchanged while an
interactive caller gets an exception it can catch instead of a dead kernel.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Sequence

# "## [0.1.0] - 2026-08-01", capturing the version. Also matches a bare
# "## [Unreleased]", which is deliberately not a valid release target.
_HEADING = re.compile(r"^##\s+\[(?P<version>[^\]]+)\]")

DEFAULT_CHANGELOG = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"


class ChangelogError(Exception):
    """A changelog that cannot be released as it stands.

    Raised by the library functions here. 'main' converts it into a non-zero
    exit carrying the same message, which is what the publish workflow gates
    on; callers that are not a command line - a notebook, a test - catch it
    instead.
    """


def running_in_kernel() -> bool:
    """Report whether this is executing inside a Jupyter/IPython kernel.

    Running a file top-to-bottom in an interactive window executes its
    ``__main__`` block as well - ``__name__`` really is ``'__main__'`` there.
    Argparse then parses the *kernel's* argv ("ipykernel_launcher.py -f
    kernel-1234.json") and aborts with ``SystemExit: 2`` before anything
    useful happens, which is confusing precisely because nothing the user
    wrote is at fault. The guard makes running the file in a kernel do what
    was meant: define the functions and say how to call them.

    Returns
    -------
    bool
        True in a kernel (``ipykernel`` is imported), False for a plain
        ``python tools/changelog.py`` - including in CI, where the publish
        workflow relies on the command line still working.
    """
    return "ipykernel" in sys.modules


def utf8_stdout() -> None:
    """Force stdout to UTF-8, whatever the console's code page is.

    The notes contain em-dashes and the like. Windows consoles default to
    cp1252, which mangles them (or raises outright) when the output is
    redirected into the release body.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


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
    ChangelogError
        If the version has no section, or the section is empty.
    """
    version = version.lstrip("vV")
    sections = parse_sections(changelog.read_text(encoding="utf-8"))

    if version not in sections:
        released = [v for v in sections if v.lower() != "unreleased"]
        hint = ""
        if sections.get("Unreleased", "").strip():
            # By far the likeliest cause: the notes were written but the
            # heading was never renamed, so they are unreleasable as they are.
            # Say exactly what to change - this message is what the person
            # cutting the release sees in the failed job's log.
            hint = (
                f"\n\n'## [Unreleased]' has notes waiting - rename that "
                f"heading in CHANGELOG.md:\n"
                f"    ## [Unreleased]        <- keep, now empty\n"
                f"    ## [{version}] - YYYY-MM-DD\n"
                f"and update the '[Unreleased]:' / '[{version}]:' links at the "
                f"foot of the file. Push that commit, then delete this tag and "
                f"re-create the release on the new commit - deleting a release "
                f"leaves its tag behind, pointing at the old commit."
            )
        raise ChangelogError(
            f"CHANGELOG.md has no '## [{version}]' section.\n"
            f"Add one before tagging. Sections present: {released}{hint}"
        )

    body = sections[version]
    if not body:
        raise ChangelogError(
            f"CHANGELOG.md section '## [{version}]' is empty."
        )
    return body


def main(argv: Sequence[str] | None = None) -> None:
    """Print the release notes for the requested version.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Arguments to parse. The default is None, meaning ``sys.argv[1:]``.
        Pass an explicit list to call this from a notebook, where ``sys.argv``
        holds the kernel's own arguments and argparse rejects them.

    Raises
    ------
    SystemExit
        With the 'ChangelogError' message when the notes cannot be extracted.
        The publish workflow gates on that non-zero exit.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="release version, e.g. 0.1.0 or v0.1.0")
    parser.add_argument(
        "--changelog",
        type=pathlib.Path,
        default=DEFAULT_CHANGELOG,
        help="path to CHANGELOG.md (default: repository root)",
    )
    args = parser.parse_args(argv)
    utf8_stdout()
    try:
        body = notes_for(args.version, args.changelog)
    except ChangelogError as exc:
        sys.exit(str(exc))
    sys.stdout.write(body + "\n")


if __name__ == "__main__":
    if running_in_kernel():
        print(
            "changelog.py loaded - the CLI is skipped in a kernel, where "
            "argparse would parse ipykernel's own argv. Call the functions "
            "instead:\n"
            "    notes_for('0.2.0')          # the release body for a version\n"
            "    parse_sections(...)         # {version: notes}\n"
            "    main(['0.2.0'])             # the CLI, with explicit argv"
        )
    else:
        main()
