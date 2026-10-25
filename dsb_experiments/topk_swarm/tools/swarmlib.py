#!/usr/bin/env python3
"""Shared helpers for the topk_swarm node tools (lease.py, append_locked.py, broker.py).

Standard library only. Python 3.9 compatible. All times are UTC ISO strings
("2026-10-25T10:00:00Z") in the registry.
"""
import datetime
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

SWARM = Path(__file__).resolve().parent.parent
STATE = SWARM / "state"
REGISTRY = STATE / "nodes.json"
LOCK = STATE / "nodes.lock"
LOGS = STATE / "logs"
RESULTS = STATE / "results"
CAP = 6
UTC = datetime.timezone.utc
ACTIVE_STATES = ("requested", "bootstrapping", "ready", "expiring")
LEASABLE_STATES = ("ready",)


def utcnow():
    return datetime.datetime.now(UTC).replace(microsecond=0)


def iso(dt):
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    if not s:
        return None
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def parse_duration(s):
    """'45m', '2h', '90s', '1h30m'. A bare number is minutes. Returns seconds."""
    s = str(s).strip().lower()
    if re.fullmatch(r"\d+", s):
        return int(s) * 60
    total = 0
    for num, unit in re.findall(r"(\d+)([smhd])", s):
        total += int(num) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    if total == 0 and s not in ("0", "0s", "0m"):
        raise ValueError("bad duration: %r" % s)
    return total


def fmt_remaining(seconds):
    if seconds is None:
        return "-"
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(seconds))
    return "%s%d:%02d" % (sign, seconds // 3600, (seconds % 3600) // 60)


def remaining_seconds(node, now=None):
    exp = parse_iso(node.get("lease_expires"))
    if exp is None:
        return None
    return (exp - (now or utcnow())).total_seconds()


def empty_registry():
    return {"cap": CAP, "updated": iso(utcnow()), "nodes": []}


def load_registry():
    if not REGISTRY.exists():
        return empty_registry()
    with open(REGISTRY) as f:
        reg = json.load(f)
    reg.setdefault("cap", CAP)
    reg.setdefault("nodes", [])
    return reg


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_registry(reg):
    reg["updated"] = iso(utcnow())
    atomic_write_text(REGISTRY, json.dumps(reg, indent=2) + "\n")


class FileLock:
    """Exclusive fcntl.flock on a lock file (macOS has no flock binary)."""

    def __init__(self, path=None):
        self.path = Path(path) if path else LOCK
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()
        return False


class Registry:
    """Locked registry access.

        with Registry() as reg:
            ... read or edit reg.data ...
            reg.save()          # write back atomically on exit (only if called)
    """

    def __init__(self):
        self.data = None
        self._dirty = False
        self._lock = FileLock(LOCK)

    def __enter__(self):
        self._lock.__enter__()
        self.data = load_registry()
        return self

    def save(self):
        self._dirty = True

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._dirty and exc_type is None:
                save_registry(self.data)
        finally:
            self._lock.__exit__(exc_type, exc, tb)
        return False


def node_label(node):
    return node.get("name") or node.get("ssh") or ("job%s" % node.get("job_id"))
