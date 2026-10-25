#!/usr/bin/env python3
"""GPU lease tool for the topk_swarm node registry (see nodes/README.md).

  python3 tools/lease.py acquire --holder <name> --arch any|b200|gb200 [--node <name>] [--min-remaining 45m] [--wait 30m]
  python3 tools/lease.py release --holder <name>
  python3 tools/lease.py reserve --holder <name> --node <node> [--for 45m]
  python3 tools/lease.py unreserve --holder <name>
  python3 tools/lease.py status

reserve marks a node for a holder who is building there before taking the measurement lease:
it queues the holder on the node (head of the line if first), so the broker does not release the
node to make room and nobody else takes the GPU in the meantime. The reservation expires after
--for, is replaced by the holder's acquire, and is removed by release or unreserve.

acquire prints one line on success and exits 0:
  node=<ssh> gpu=<index> root=<root> exec_prefix=<prefix> arch=<arch> cuda=<cuda> expires=<iso> login=<login-host>
(exec_prefix is shell-quoted, parse the line with shlex.split). If the same holder already
holds a matching GPU, that lease is printed again (idempotent). With nothing free it polls
the registry until --wait elapses, then exits 2. Exit 1 on usage or registry errors.

Only gpus[].holder, gpus[].since and gpus[].queue are changed, under fcntl.flock on
state/nodes.lock, with an atomic write of state/nodes.json.

Node-specific waits are first come first served: `acquire --node X --wait T` registers the
holder in that node's gpus[].queue (entry expires after T). While a queue is non-empty, only
its head may take the GPU, plain acquires skip it. The entry is removed on success or timeout.
"""
import argparse
import os
import shlex
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import datetime  # noqa: E402

from swarmlib import (LEASABLE_STATES, LOGS, Registry, fmt_remaining, iso, parse_duration,  # noqa: E402
                      parse_iso, remaining_seconds, utcnow)

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


def prune_queue(gpu, now):
    """Drop expired waiters. Returns (queue, changed)."""
    old = gpu.get("queue") or []
    q = [w for w in old if parse_iso(w.get("until")) is not None and parse_iso(w["until"]) > now]
    changed = len(q) != len(old)
    if q:
        gpu["queue"] = q
    elif "queue" in gpu:
        del gpu["queue"]
    return q, changed


def dequeue(reg, holder):
    """Remove holder from every queue. Returns True if anything changed."""
    changed = False
    for node in reg.data.get("nodes", []):
        for gpu in node.get("gpus", []):
            q = gpu.get("queue") or []
            nq = [w for w in q if w.get("holder") != holder]
            if len(nq) != len(q):
                changed = True
                if nq:
                    gpu["queue"] = nq
                else:
                    del gpu["queue"]
    return changed


def try_acquire(reg, holder, arch, min_remaining_s, want_node=None, wait_until=None):
    """Returns (node, gpu) or None. Mutates reg.data and calls reg.save() when anything changed.
    want_node restricts the choice to one node (name or ssh host), for reruns on a card's node.
    With want_node and wait_until the holder is queued on that node, first come first served."""
    now = utcnow()
    candidates = []
    changed = False
    for node in reg.data.get("nodes", []):
        if node.get("status") not in LEASABLE_STATES:
            continue
        if arch != "any" and node.get("arch") != arch:
            continue
        if want_node and want_node not in (node.get("name"), node.get("ssh")):
            continue
        rem = remaining_seconds(node, now)
        if rem is None or rem < min_remaining_s:
            continue
        for gpu in node.get("gpus", []):
            if gpu.get("holder") == holder:
                if dequeue(reg, holder):
                    reg.save()
                return node, gpu  # already held, idempotent
        if node.get("retire"):
            queued_here = any(w.get("holder") == holder for g in node.get("gpus", []) for w in (g.get("queue") or []))
            if not queued_here:
                continue  # retiring node: only waiters already in its queue may still take it
        if node.get("dedicated_to") and holder not in node["dedicated_to"]:
            continue  # reserved for the final evaluation holders
        for gpu in node.get("gpus", []):
            q, pruned = prune_queue(gpu, now)
            changed = changed or pruned
            if want_node and wait_until is not None and not any(w.get("holder") == holder for w in q):
                q.append({"holder": holder, "since": iso(now), "until": iso(wait_until)})
                gpu["queue"] = q
                changed = True
            if gpu.get("holder"):
                continue
            if q and q[0].get("holder") != holder:
                continue  # reserved for the first waiter
            candidates.append((rem, node, gpu))
            break
    if not candidates:
        if changed:
            reg.save()
        return None
    rem, node, gpu = max(candidates, key=lambda c: c[0])
    gpu["holder"] = holder
    gpu["since"] = iso(now)
    dequeue(reg, holder)
    reg.save()
    return node, gpu


def cmd_acquire(args):
    min_remaining_s = parse_duration(args.min_remaining)
    wait_s = parse_duration(args.wait)
    deadline = time.monotonic() + wait_s
    wait_until = (utcnow() + datetime.timedelta(seconds=wait_s + 60)) if (args.node and wait_s > 0) else None
    first = True
    while True:
        with Registry() as reg:
            got = try_acquire(reg, args.holder, args.arch, min_remaining_s, args.node, wait_until)
            if got is None:
                nodes = reg.data.get("nodes", [])
        if got is not None:
            node, gpu = got
            audit("acquire", args.holder, "node=%s gpu=%s arch=%s since=%s" % (node.get("ssh"), gpu.get("index"), args.arch, gpu.get("since")))
            print(lease_line(node, gpu))
            return 0
        if first:
            ready = [n for n in nodes if n.get("status") in LEASABLE_STATES]
            sys.stderr.write("lease: no free %s GPU%s with >= %s remaining (%d ready node(s)), waiting up to %s\n"
                             % (args.arch, (" on " + args.node) if args.node else "", args.min_remaining, len(ready), args.wait))
            first = False
        if time.monotonic() >= deadline:
            if wait_until is not None:
                with Registry() as reg:
                    if dequeue(reg, args.holder):
                        reg.save()
            audit("acquire-timeout", args.holder, "arch=%s node=%s wait=%s" % (args.arch, args.node, args.wait))
            sys.stderr.write("lease: nothing free after %s, giving up\n" % args.wait)
            return 2
        time.sleep(min(POLL_SECONDS, max(1, deadline - time.monotonic())))


def cmd_reserve(args):
    until = utcnow() + datetime.timedelta(seconds=parse_duration(args.for_))
    with Registry() as reg:
        found = None
        for node in reg.data.get("nodes", []):
            if args.node in (node.get("name"), node.get("ssh")) and node.get("status") in LEASABLE_STATES:
                found = node
        if found is None:
            sys.stderr.write("lease: no ready node named %s\n" % args.node)
            return 1
        dequeue(reg, args.holder)
        for gpu in found.get("gpus", []):
            q = gpu.get("queue") or []
            q.append({"holder": args.holder, "since": iso(utcnow()), "until": iso(until), "reserve": True})
            gpu["queue"] = q
        reg.save()
    audit("reserve", args.holder, "node=%s until=%s" % (found.get("ssh"), iso(until)))
    print("reserved %s for %s until %s (position %d)" % (found.get("ssh"), args.holder, iso(until), len(found["gpus"][0]["queue"])))
    return 0


def cmd_unreserve(args):
    with Registry() as reg:
        changed = dequeue(reg, args.holder)
        if changed:
            reg.save()
    audit("unreserve", args.holder, "changed=%s" % changed)
    print("unreserved %s: %s" % (args.holder, "done" if changed else "nothing queued"))
    return 0


def cmd_release(args):
    freed = []
    with Registry() as reg:
        for node in reg.data.get("nodes", []):
            for gpu in node.get("gpus", []):
                if gpu.get("holder") == args.holder:
                    gpu["holder"] = None
                    gpu["since"] = None
                    freed.append("%s:gpu%s" % (node.get("ssh") or node.get("name"), gpu.get("index")))
        if dequeue(reg, args.holder):
            freed = freed or ["(reservation only)"]
            reg.save()
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
        gpus = ", ".join("gpu%s=%s%s" % (g.get("index"), g.get("holder") or "free",
                                          (" (queue: %s)" % ",".join(w["holder"] for w in g["queue"])) if g.get("queue") else "")
                         for g in node.get("gpus", [])) or "-"
        if node.get("retire"):
            gpus += "  [retiring: %s]" % node["retire"][:40]
        if node.get("dedicated_to"):
            gpus += "  [dedicated: %s]" % ",".join(node["dedicated_to"])
        print(fmt % (node.get("name"), node.get("cluster"), node.get("arch"), node.get("status"),
                     node.get("job_id"), fmt_remaining(remaining_seconds(node, now)), gpus))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire", help="take a free GPU lease", allow_abbrev=False)
    a.add_argument("--holder", required=True)
    a.add_argument("--arch", default="any", choices=["any", "b200", "gb200"])
    a.add_argument("--node", default=None, help="only this node (name or ssh host), e.g. to rerun on a card's node")
    a.add_argument("--min-remaining", default="45m", help="minimum lease time left (default 45m)")
    a.add_argument("--wait", default="30m", help="how long to wait for a free GPU (default 30m, 0 for none)")
    a.set_defaults(func=cmd_acquire)
    r = sub.add_parser("release", help="free every GPU held by --holder", allow_abbrev=False)
    r.add_argument("--holder", required=True)
    r.set_defaults(func=cmd_release)
    rv = sub.add_parser("reserve", help="mark a node for a holder who is building there", allow_abbrev=False)
    rv.add_argument("--holder", required=True)
    rv.add_argument("--node", required=True)
    rv.add_argument("--for", dest="for_", default="45m")
    rv.set_defaults(func=cmd_reserve)
    uv = sub.add_parser("unreserve", help="drop a holder's reservations", allow_abbrev=False)
    uv.add_argument("--holder", required=True)
    uv.set_defaults(func=cmd_unreserve)
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
