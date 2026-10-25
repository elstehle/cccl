#!/usr/bin/env python3
"""Generate the phase 3 node script from harness/phase3_adapters.json.

  python3 tools/phase3.py --node <ssh host> --root <on-node root> --exec-prefix "<prefix>" --gpu <i> \
      --arch b200 --pass random --out state/phase3/phase3_b200_random.sh [--scopes block,warp,subwarp]
  python3 tools/phase3.py ... --pass tie_heavy --cell-list state/phase3/close_cells_b200.json --out ...
  python3 tools/phase3.py ... --pass gb200 --arch gb200 --out ...
  python3 tools/phase3.py ... --pass 3b --out ...

The script runs on the Mac and drives the node over ssh (same pattern as every batch so far): ship
the harness and the approach sources with SOURCE_COMMIT, build every batch outside the lock,
acquire the pinned lease, run the batches in order (each run.py call is one lock section with its
own canaries), fetch the results to state/results/eval-engineer/phase3/<arch>/ after every
adapter, release the lease. Nothing is executed by this tool, it only writes the script.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)
REQS = ["not_guaranteed", "prefer_smaller_index"]


def swarm_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=SWARM).stdout.strip()
    except Exception:
        return "unknown"


def label_of(a):
    return a["id"] + (("_" + a["label"]) if a.get("label") else "")


def runpy_args(a, scope_cfg, cells_arg, req, args, extra_regimes=None, control=None):
    """One run.py invocation (without --build-only or --measure-only)."""
    second = control is not None
    control = control or scope_cfg["control"]
    out = ["--adapter", a["id"], "--control", control, "--req", req]
    if not second and req == "prefer_smaller_index" and not scope_cfg.get("control_supports_prefer_smaller_index", True):
        out += ["--control-req", "not_guaranteed"]
    out += cells_arg
    if a.get("threads"):
        out += ["--threads", str(a["threads"])]
    if a.get("n"):
        out += ["--n", str(a["n"])]
    if a.get("k"):
        out += ["--k", str(a["k"])]
    if a.get("runs"):
        out += ["--runs", str(a["runs"])]
    regimes = extra_regimes or a.get("regimes")
    if regimes:
        out += ["--regimes", regimes]
    out += ["--patterns", args.patterns, "--seeds", str(args.seeds)]
    if args.check_sorted:
        out.append("--check-sorted")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(SWARM, "harness", "phase3_adapters.json"))
    ap.add_argument("--node", required=True)
    ap.add_argument("--root", default="/work/topk_swarm")
    ap.add_argument("--host-root", help="host path of the root when the container mounts it (GB200 enroot), "
                    "used for tar in and out; default: same as --root (docker nodes)")
    ap.add_argument("--exec-prefix", required=True, help="e.g. 'docker exec -i topk_swarm' or the enroot start line")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arch", required=True, choices=["b200", "gb200"])
    ap.add_argument("--holder", default="eval-engineer")
    ap.add_argument("--pass", dest="pass_", default="random", choices=["random", "tie_heavy", "gb200", "3b"])
    ap.add_argument("--scopes", default="block,warp,subwarp", help="order of the scopes")
    ap.add_argument("--cell-list", help="tie_heavy pass: the close cells file written by tools/winners.py")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--min-remaining", default="3h")
    ap.add_argument("--only", default="all", choices=["all", "build", "measure"],
                    help="write only the ship and build steps, or only the lease, measure and fetch steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    host_root = args.host_root or args.root
    is_docker = args.exec_prefix.strip().startswith("docker")
    # how to run a command inside the container and how to move files in and out
    def inside(cmd):
        return "ssh -o BatchMode=yes %s %s" % (args.node, shlex.quote(args.exec_prefix + " bash -c " + shlex.quote(cmd)))

    def tar_in(paths):
        if is_docker:
            return "tar czf - -C %s %s | ssh -o BatchMode=yes %s %s" % (
                SWARM, " ".join(paths), args.node, shlex.quote(args.exec_prefix + " tar xzf - -C %s/work/%s" % (args.root, args.holder)))
        return "tar czf - -C %s %s | ssh -o BatchMode=yes %s %s" % (
            SWARM, " ".join(paths), args.node, shlex.quote("tar xzf - -C %s/work/%s" % (host_root, args.holder)))

    def tar_out(dest):
        if is_docker:
            return "ssh -o BatchMode=yes %s %s | tar xzf - -C %s" % (
                args.node, shlex.quote(args.exec_prefix + " tar czf - -C %s/results/%s ." % (args.root, args.holder)), dest)
        return "ssh -o BatchMode=yes %s %s | tar xzf - -C %s" % (
            args.node, shlex.quote("tar czf - -C %s/results/%s ." % (host_root, args.holder)), dest)

    args.patterns = "tie_heavy" if args.pass_ == "tie_heavy" else "random"
    args.check_sorted = args.pass_ == "3b"
    scopes = args.scopes.split(",")
    if args.pass_ == "gb200":
        scopes = [s for s in m["gb200_pass"]["scopes"] if s in scopes]

    # batches: list of (scope, adapter dict, cells args list, req)
    batches = []
    ship = set()
    for scope in scopes:
        sc = m["scopes"][scope]
        ship.update(sc.get("ship", []))
        if args.pass_ == "3b":
            entries = m["phase3b"]["adapters"].get(scope, [])
            adapters = [e if isinstance(e, dict) else {"id": e} for e in entries]
            cells_names = m["phase3b"]["cells"]
        else:
            adapters = sc["adapters"]
            cells_names = None
        for a in adapters:
            a = dict(a)
            if args.pass_ == "gb200" and scope == "block":
                a["n"] = m["gb200_pass"]["block_n"]
            if args.pass_ == "tie_heavy":
                if not args.cell_list:
                    raise SystemExit("--cell-list is required for the tie_heavy pass")
                cells_sets = [["--cell-list", "cells/" + os.path.basename(args.cell_list)]]
            else:
                names = cells_names or a.get("cells") or list(m["cells"].keys())
                cells_sets = []
                for nm in names:
                    path = m["cells"][nm]
                    if nm == "named":
                        cells_sets.append(["--named-cells", "cells/" + os.path.basename(path)])
                    else:
                        cells_sets.append(["--cells", "cells/" + os.path.basename(path)])
            for cells_arg in cells_sets:
                for req in REQS:
                    batches.append((scope, a, cells_arg, req, None))
                    second = sc.get("psi_second_control")
                    if second and req == "prefer_smaller_index" and args.pass_ in ("random", "tie_heavy"):
                        batches.append((scope, a, cells_arg, req, second))

    label = "phase3_%s_%s" % (args.arch, args.pass_)
    dest = os.path.join(SWARM, "state", "results", args.holder, "phase3", args.arch)
    lines = ["#!/bin/bash", "# generated by tools/phase3.py, pass %s, arch %s, node %s" % (args.pass_, args.arch, args.node),
             "set -u", "SWARM=%s" % shlex.quote(SWARM), "mkdir -p %s" % shlex.quote(dest), ""]
    build_lines = []
    if args.only in ("all", "build"):
        lines.append("# 1. source commit and shipping (harness, cells files, approach sources the adapters include)")
        lines.append("git -C $SWARM rev-parse HEAD > $SWARM/harness/SOURCE_COMMIT")
        lines.append("mkdir -p $SWARM/harness/cells && cp $SWARM/harness/cells_*.json $SWARM/ideas/named_cells.json $SWARM/harness/cells/")
        if args.cell_list:
            lines.append("cp %s $SWARM/harness/cells/" % shlex.quote(args.cell_list))
        lines.append(inside("mkdir -p %s/work/%s %s/results/%s" % (args.root, args.holder, args.root, args.holder)))
        lines.append(tar_in(["harness"] + sorted(ship)))
        lines.append("")
        lines.append("# 2. build everything outside the lock")
    runpy = "cd %s/work/%s/harness && export PATH=/usr/local/cuda/bin:$PATH && python3 run.py --repo %s/cccl --root %s --holder %s" % (
        args.root, args.holder, args.root, args.root, args.holder)
    if args.only in ("all", "build"):
        for scope, a, cells_arg, req, ctrl in batches:
            cmd = runpy + " " + " ".join(shlex.quote(x) for x in runpy_args(a, m["scopes"][scope], cells_arg, req, args, control=ctrl)) + \
                  " --build-only --label %s --jobs %d 2>&1 | tail -n 1" % (label, args.jobs)
            lines.append(inside(cmd))
        lines.append("")
    if args.only == "build":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        os.chmod(args.out, 0o755)
        print("wrote %s: build steps for %d batches" % (args.out, len(batches)))
        return 0
    lines.append("# 3. pinned lease, canary, batches in order, fetch after every adapter, canary, release")
    lines.append("L=$(python3 $SWARM/tools/lease.py acquire --holder %s --arch %s --node %s --min-remaining %s --wait 60m | tail -1) || exit 1" % (
        args.holder, args.arch, args.node, args.min_remaining))
    lines.append("echo \"$L\"")
    last_label = None
    for scope, a, cells_arg, req, ctrl in batches:
        tag = "%s_%s_%s%s" % (label, label_of(a), req, ("_vs_" + ctrl) if ctrl else "")
        cmd = runpy + " --gpu %d --node-name %s --arch-name %s " % (args.gpu, args.node, args.arch) + \
              " ".join(shlex.quote(x) for x in runpy_args(a, m["scopes"][scope], cells_arg, req, args, control=ctrl)) + \
              " --measure-only --label %s > %s/work/%s/%s.log 2>&1; grep -h 'cells with\\|FAILED\\|WARNING' %s/work/%s/%s.log | cut -c1-160" % (
                  label, args.root, args.holder, tag, args.root, args.holder, tag)
        lines.append("echo '== %s %s %s %s%s'" % (scope, label_of(a), " ".join(cells_arg), req, (" control " + ctrl) if ctrl else ""))
        lines.append(inside(cmd))
        if label_of(a) != last_label:
            lines.append(tar_out(dest))
            last_label = label_of(a)
    lines.append(tar_out(dest))
    lines.append("python3 $SWARM/tools/lease.py release --holder %s | tail -1" % args.holder)
    lines.append("")
    lines.append("# 4. winners matrix from the fetched batches")
    seconds = sorted({sc.get("psi_second_control") for sc in m["scopes"].values() if sc.get("psi_second_control")})
    lines.append("python3 $SWARM/tools/winners.py --results %s --arch %s --out $SWARM/WINNERS_%s.md --json $SWARM/state/phase3/winners_%s.json --close $SWARM/state/phase3/close_cells_%s.json%s" % (
        shlex.quote(dest), args.arch, args.arch, args.arch, args.arch,
        "".join(" --second-reference %s" % x for x in seconds)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    os.chmod(args.out, 0o755)
    print("wrote %s: %d batches (%s), ships %s, head %s" % (args.out, len(batches), ", ".join(scopes), sorted(ship), swarm_head()[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
