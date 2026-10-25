#!/usr/bin/env python3
"""Append an entry to a shared markdown file under the swarm lock.

  python3 tools/append_locked.py <shared-md> <entry-file>

<shared-md> is absolute, relative to the current directory, or relative to $SWARM
(so `TECHNIQUES.md` works from anywhere). The entry is appended after one blank line,
with an atomic write, under fcntl.flock on state/nodes.lock (the same lock the registry
uses). When the target is TECHNIQUES.md and the entry heading is `### T?: <name>`, the `?`
is replaced by the next free id found by scanning existing `### T<n>:` headings.
Prints `appended <file> [as T<n>]` and exits 0, exits 1 on errors.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarmlib import SWARM, FileLock, atomic_write_text  # noqa: E402

HEADING_PLACEHOLDER = re.compile(r"^(#{2,4}\s+)T\?:", re.M)
HEADING_ID = re.compile(r"^#{2,4}\s+T(\d+):", re.M)


def resolve_target(arg):
    p = Path(arg)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (SWARM / p).resolve()


def next_id(text):
    ids = [int(m.group(1)) for m in HEADING_ID.finditer(text)]
    return (max(ids) + 1) if ids else 1


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 1
    target = resolve_target(argv[1])
    entry_path = Path(argv[2])
    if not target.exists():
        sys.stderr.write("append_locked: target does not exist: %s\n" % target)
        return 1
    if not entry_path.exists():
        sys.stderr.write("append_locked: entry file does not exist: %s\n" % entry_path)
        return 1
    entry = entry_path.read_text().strip("\n")
    if not entry.strip():
        sys.stderr.write("append_locked: entry is empty\n")
        return 1
    assigned = None
    with FileLock():
        text = target.read_text()
        if target.name == "TECHNIQUES.md" and HEADING_PLACEHOLDER.search(entry):
            assigned = next_id(text)
            entry = HEADING_PLACEHOLDER.sub(lambda m: "%sT%d:" % (m.group(1), assigned), entry, count=1)
        if text and not text.endswith("\n"):
            text += "\n"
        if text and not text.endswith("\n\n"):
            text += "\n"
        atomic_write_text(target, text + entry + "\n")
    print("appended %s%s" % (target, (" as T%d" % assigned) if assigned else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
