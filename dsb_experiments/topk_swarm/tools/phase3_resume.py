#!/usr/bin/env python3
"""Resume the phase 3 passes after the 2026-10-25 21:26Z network outage, in one command.

  python3 tools/phase3_resume.py [--dry-run] [--no-network-check] [--arches b200,gb200]

Case A (a node dedicated to eval-engineer with enough lease is ready for the arch): reuse it, the
builds skip existing binaries, the measure chains skip batches already measured on it.
Case B (none): request one through `tools/broker.py request --cluster <c> --dedicate
eval-engineer,hw-researcher --make-room --wait 30m`, bootstrap it, wait until ready, then build from
scratch (skip-existing finds nothing) and run the remaining batches there. Every batch measured on a
node other than the original pass node carries `note: resumed on <node> ...` in its meta, and
tools/winners.py prints the node per cell and per table.

Order per arch, the two arch tracks run concurrently:
  gb200: warp and sub-warp pass, named sub-warp addendum, block cells at N 1024 and 4096
  b200 : tie_heavy pass (close cells), B1_det3 addendum, named sub-warp addendum, phase 3b
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)
STATE = os.path.join(SWARM, "state")
P3 = os.path.join(STATE, "phase3")
LOGS = os.path.join(P3, "logs")
HOLDER = "eval-engineer"
ORIGINAL = {"b200": "umbriel-b200-035", "gb200": "gb200-nvl4-ts2-105"}
CLUSTER = {"b200": "computelab", "gb200": "dlcluster"}
PY = sys.executable or "python3"


def log(msg):
    sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%SZ", time.gmtime()), msg))
    sys.stdout.flush()


def network_up():
    try:
        socket.gethostbyname("computelab-sc-01")
        return True
    except Exception:
        return False


def registry():
    return json.load(open(os.path.join(STATE, "nodes.json")))


def remaining_min(node):
    exp = node.get("lease_expires")
    if not exp:
        return 0
    try:
        t = time.mktime(time.strptime(exp.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except Exception:
        return 0
    return (t - time.time()) / 60.0


def dedicated_ready(arch, min_remaining):
    best = None
    for n in registry().get("nodes", []):
        if n.get("arch") != arch or n.get("status") != "ready":
            continue
        ded = n.get("dedicated") or n.get("dedicated_to") or []
        if isinstance(ded, str):
            ded = [x.strip() for x in ded.split(",")]
        if HOLDER not in ded:
            continue
        if remaining_min(n) < min_remaining:
            continue
        if best is None or n.get("name") == ORIGINAL.get(arch) or remaining_min(n) > remaining_min(best):
            best = n
    return best


def dedicated_pending(arch):
    """A dedicated successor the broker already requested (requested or bootstrapping): wait, do not double-request."""
    for n in registry().get("nodes", []):
        if n.get("arch") != arch or n.get("status") not in ("requested", "bootstrapping"):
            continue
        ded = n.get("dedicated") or n.get("dedicated_to") or []
        if isinstance(ded, str):
            ded = [x.strip() for x in ded.split(",")]
        if HOLDER in ded:
            return n
    return None


def request_dedicated(arch, dry):
    cmd = [PY, os.path.join(HERE, "broker.py"), "request", "--cluster", CLUSTER[arch], "--dedicate",
           "eval-engineer,hw-researcher", "--make-room", "--wait", "30m",
           "--note", "phase 3 resume after the 21:26Z outage"]
    log("requesting a dedicated %s node: %s" % (arch, " ".join(cmd)))
    if dry:
        return
    subprocess.run(cmd)


def wait_dedicated(arch, min_remaining, timeout_s, dry):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        n = dedicated_ready(arch, min_remaining)
        if n:
            return n
        # bootstrap anything of ours that is sitting in requested or bootstrapping state
        for m in registry().get("nodes", []):
            ded = m.get("dedicated") or m.get("dedicated_to") or []
            if isinstance(ded, str):
                ded = [x.strip() for x in ded.split(",")]
            if m.get("arch") == arch and HOLDER in ded and m.get("status") in ("requested",) and m.get("name") \
                    and not str(m.get("name")).startswith("pending"):
                cmd = [PY, os.path.join(HERE, "broker.py"), "bootstrap", m["name"]]
                log("bootstrapping %s" % m["name"])
                if not dry:
                    subprocess.run(cmd)
        if dry:
            return None
        time.sleep(60)
    return None


def host_root_of(node):
    pref = node.get("exec_prefix") or ""
    root = node.get("root") or "/work/topk_swarm"
    if pref.startswith("docker"):
        return root
    # enroot: "-m <host>:<container>" maps the container root onto the host
    toks = pref.split()
    for i, tk in enumerate(toks):
        if tk == "-m" and i + 1 < len(toks) and ":" in toks[i + 1]:
            h, c = toks[i + 1].split(":", 1)
            if root.startswith(c):
                return h + root[len(c):]
    return root


def gen(node, arch, pass_, extra, name, only, split=None, no_ship=False, wait=False, note="", jobs=None):
    out = os.path.join(P3, name + ".sh")
    cmd = [PY, os.path.join(HERE, "phase3.py"), "--node", node["ssh"] or node["name"], "--root", node.get("root", "/work/topk_swarm"),
           "--host-root", host_root_of(node), "--exec-prefix", node["exec_prefix"], "--arch", arch, "--pass", pass_,
           "--only", only, "--min-remaining", "10m", "--out", out] + extra
    if split:
        cmd += ["--split", split]
    if no_ship:
        cmd.append("--no-ship")
    if wait:
        cmd.append("--wait-for-build")
    if note:
        cmd += ["--note", note]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("phase3.py failed: %s %s" % (r.stdout.strip()[-200:], r.stderr.strip()[-400:]))
        return None
    return out


def run_pass(node, arch, pass_, extra, tag, chains, jobs, note, dry):
    """Generate build chains and the measure script for one pass and run them (measure blocks)."""
    builds = []
    for i in range(1, chains + 1):
        b = gen(node, arch, pass_, extra, "%s_build_%dof%d" % (tag, i, chains), "build", split="%d/%d" % (i, chains),
                no_ship=(i > 1), note=note, jobs=jobs)
        if b:
            builds.append(b)
    meas = gen(node, arch, pass_, extra, "%s_measure" % tag, "measure", wait=True, note=note)
    if not meas or not builds:
        log("%s: script generation failed, skipping" % tag)
        return
    log("%s on %s: %d build chains, measure script %s" % (tag, node["name"], len(builds), os.path.basename(meas)))
    if dry:
        return
    procs = []
    for i, b in enumerate(builds):
        if i:
            time.sleep(90)  # the first chain ships the sources
        lf = open(os.path.join(LOGS, os.path.basename(b)[:-3] + ".log"), "a")
        procs.append(subprocess.Popen(["bash", b], stdout=lf, stderr=subprocess.STDOUT))
    time.sleep(30)
    lf = open(os.path.join(LOGS, os.path.basename(meas)[:-3] + ".log"), "a")
    rc = subprocess.run(["bash", meas], stdout=lf, stderr=subprocess.STDOUT).returncode
    for p in procs:
        p.wait()
    log("%s done (measure rc %d)" % (tag, rc))


def track(arch, args):
    node = dedicated_ready(arch, 25)
    if not node:
        pending = dedicated_pending(arch)
        if pending:
            log("%s: the broker's successor %s is %s, waiting for it instead of requesting" % (
                arch, pending.get("name"), pending.get("status")))
        elif args.no_request:
            log("%s: no dedicated node ready and requests disabled" % arch)
            return
        else:
            request_dedicated(arch, args.dry_run)
        node = wait_dedicated(arch, 60, 45 * 60, args.dry_run)
        if not node:
            log("%s: no dedicated node became ready, giving up this track" % arch)
            return
    note = ""
    if node["name"] != ORIGINAL[arch]:
        note = ("resumed on %s after the 2026-10-25 21:26Z network outage, the first batches of this pass ran on %s"
                % (node["name"], ORIGINAL[arch]))
    log("%s track on %s (%.0f min left)%s" % (arch, node["name"], remaining_min(node), " [resumed on a new node]" if note else ""))
    if arch == "gb200":
        chains, jobs = 4, 32
        run_pass(node, arch, "gb200", ["--scopes", "warp,subwarp"], "r_gb200_warp", chains, jobs, note, args.dry_run)
        run_pass(node, arch, "random", ["--scopes", "subwarp", "--cells-only", "named"], "r_gb200_named_sub", chains, jobs, note, args.dry_run)
        run_pass(node, arch, "gb200", ["--scopes", "block"], "r_gb200_block", chains, jobs, note, args.dry_run)
    else:
        chains, jobs = 2, 14
        close = os.path.join(P3, "close_cells_b200.json")
        run_pass(node, arch, "tie_heavy", ["--scopes", "block,warp,subwarp", "--cell-list", close], "r_b200_tie", chains, jobs, note, args.dry_run)
        run_pass(node, arch, "random", ["--scopes", "block", "--adapters", "B1_det3,B1_det3_runs4"], "r_b200_det3", chains, jobs, note, args.dry_run)
        run_pass(node, arch, "random", ["--scopes", "subwarp", "--cells-only", "named"], "r_b200_named_sub", chains, jobs, note, args.dry_run)
        run_pass(node, arch, "3b", ["--scopes", "block,warp,subwarp"], "r_b200_3b", chains, jobs, note, args.dry_run)
    log("%s track finished" % arch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="generate scripts and print the plan, run nothing")
    ap.add_argument("--no-network-check", action="store_true")
    ap.add_argument("--no-request", action="store_true", help="never request nodes, only reuse dedicated ones")
    ap.add_argument("--arches", default="gb200,b200")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if not args.no_network_check and not network_up():
        log("cluster names do not resolve, the tunnel is down, nothing started")
        return 2
    threads = [threading.Thread(target=track, args=(a, args), name=a) for a in args.arches.split(",")]
    for t in threads:
        t.start()
        time.sleep(5)
    for t in threads:
        t.join()
    log("all tracks finished, regenerate the matrices with tools/winners.py if the measure scripts did not")
    return 0


if __name__ == "__main__":
    sys.exit(main())
