#!/usr/bin/env python3
"""GPU lease tool for the topk_swarm node registry (see nodes/README.md).

  python3 tools/lease.py acquire --holder <name> --arch any|b200|gb200 [--min-remaining 45m] [--wait 30m]
  python3 tools/lease.py release --holder <name>
  python3 tools/lease.py status

acquire prints one line on success and exits 0:
  node=<ssh> gpu=<index> root=<root> exec_prefix=<prefix> arch=<arch> cuda=<cuda> expires=<iso> login=<login-host>
(exec_prefix is shell-quoted, parse the line with shlex.split). If the same holder already
holds a matching GPU, that lease is printed again (idempotent). With nothing free it polls
the registry until --wait elapses, then exits 2. Exit 1 on usage or registry errors.

Only gpus[].holder and gpus[].since are changed, under fcntl.flock on state/nodes.lock,
with an atomic write of state/nodes.json.
"""
import argparse
import os
import shlex
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from swarmlib import (LEASABLE_STATES, LOGS, Registry, fmt_remaining, iso, parse_duration,  # noqa: E402
                      remaining_seconds, utcnow)

POLL_SECONDS = 15
LEASE_LOG = LOGS / "lease.log"


def audit(event, holder, detail):
    """Append one line per acquire/release to state/logs/lease.log so a vanished lease can be traced."""
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LEASE_LOG, "a") as f:
            f.write("%s %s holder=%s pid=%d ppid=%d cwd=%s %s\n" % (
                iso(utcnow()), event, holder, os.getpid(), os.getppid(), os.getcwd(), detail))
    except OSError:
        pass


def lease_line(node, gpu):
    return ("node=%s gpu=%s root=%s exec_prefix=%s arch=%s cuda=%s expires=%s login=%s" % (
        node.get("ssh"), gpu.get("index"), node.get("root"),
        shlex.quote(node.get("exec_prefix") or ""), node.get("arch"), node.get("cuda"),
        node.get("lease_expires"), node.get("login")))


def try_acquire(reg, holder, arch, min_remaining_s):
    """Returns (node, gpu) or None. Mutates reg.data and calls reg.save() on success."""
    now = utcnow()
    candidates = []
    for node in reg.data.get("nodes", []):
        if node.get("status") not in LEASABLE_STATES:
            continue
        if arch != "any" and node.get("arch") != arch:
            continue
        rem = remaining_seconds(node, now)
        if rem is None or rem < min_remaining_s:
            continue
        for gpu in node.get("gpus", []):
            if gpu.get("holder") == holder:
                return node, gpu  # already held, idempotent
        for gpu in node.get("gpus", []):
            if not gpu.get("holder"):
                candidates.append((rem, node, gpu))
                break
    if not candidates:
        return None
    rem, node, gpu = max(candidates, key=lambda c: c[0])
    gpu["holder"] = holder
    gpu["since"] = iso(now)
    reg.save()
    return node, gpu


def cmd_acquire(args):
    min_remaining_s = parse_duration(args.min_remaining)
    wait_s = parse_duration(args.wait)
    deadline = time.monotonic() + wait_s
    first = True
    while True:
        with Registry() as reg:
            got = try_acquire(reg, args.holder, args.arch, min_remaining_s)
            if got is None:
                nodes = reg.data.get("nodes", [])
        if got is not None:
            node, gpu = got
            audit("acquire", args.holder, "node=%s gpu=%s arch=%s since=%s" % (node.get("ssh"), gpu.get("index"), args.arch, gpu.get("since")))
            print(lease_line(node, gpu))
            return 0
        if first:
            ready = [n for n in nodes if n.get("status") in LEASABLE_STATES]
            sys.stderr.write("lease: no free %s GPU with >= %s remaining (%d ready node(s)), waiting up to %s\n"
                             % (args.arch, args.min_remaining, len(ready), args.wait))
            first = False
        if time.monotonic() >= deadline:
            audit("acquire-timeout", args.holder, "arch=%s wait=%s" % (args.arch, args.wait))
            sys.stderr.write("lease: nothing free after %s, giving up\n" % args.wait)
            return 2
        time.sleep(min(POLL_SECONDS, max(1, deadline - time.monotonic())))


def cmd_release(args):
    freed = []
    with Registry() as reg:
        for node in reg.data.get("nodes", []):
            for gpu in node.get("gpus", []):
                if gpu.get("holder") == args.holder:
                    gpu["holder"] = None
                    gpu["since"] = None
                    freed.append("%s:gpu%s" % (node.get("ssh") or node.get("name"), gpu.get("index")))
        if freed:
            reg.save()
    audit("release", args.holder, "freed=%s" % (",".join(freed) if freed else "nothing"))
    print("released %s: %s" % (args.holder, ", ".join(freed) if freed else "nothing held"))
    return 0


def cmd_status(args):
    with Registry() as reg:
        data = reg.data
    now = utcnow()
    nodes = data.get("nodes", [])
    print("registry %s  cap=%s  nodes=%d  (now %s)" % (data.get("updated"), data.get("cap"), len(nodes), iso(now)))
    fmt = "%-22s %-10s %-6s %-13s %-9s %-9s %s"
    print(fmt % ("node", "cluster", "arch", "status", "job", "remaining", "gpus"))
    for node in nodes:
        if node.get("status") in ("released", "lost") and not args.all:
            continue
        gpus = ", ".join("gpu%s=%s" % (g.get("index"), g.get("holder") or "free") for g in node.get("gpus", [])) or "-"
        print(fmt % (node.get("name"), node.get("cluster"), node.get("arch"), node.get("status"),
                     node.get("job_id"), fmt_remaining(remaining_seconds(node, now)), gpus))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire", help="take a free GPU lease")
    a.add_argument("--holder", required=True)
    a.add_argument("--arch", default="any", choices=["any", "b200", "gb200"])
    a.add_argument("--min-remaining", default="45m", help="minimum lease time left (default 45m)")
    a.add_argument("--wait", default="30m", help="how long to wait for a free GPU (default 30m, 0 for none)")
    a.set_defaults(func=cmd_acquire)
    r = sub.add_parser("release", help="free every GPU held by --holder")
    r.add_argument("--holder", required=True)
    r.set_defaults(func=cmd_release)
    s = sub.add_parser("status", help="print the registry")
    s.add_argument("--all", action="store_true", help="include released and lost nodes")
    s.set_defaults(func=cmd_status)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as e:
        sys.stderr.write("lease: error: %s\n" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
