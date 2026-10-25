#!/usr/bin/env python3
"""Node broker for topk_swarm. Owns state/nodes.json (see nodes/README.md). Stdlib only.

  broker.py status [--json] [--all]              squeue on both login hosts, reconcile, mark lost/expiring
  broker.py request --cluster computelab|dlcluster [--count 1] [--wait 0]
                                                 detachable sbatch hold (4 h), respects the cap of 6,
                                                 excludes hosts that already carry an active entry
  broker.py bootstrap <node-or-job> [--image REF] run nodes/bootstrap_<cluster>.sh on the node, mark ready
  broker.py release <node-or-job> [--force]      remove the container, scancel, mark released
  broker.py retire <node-or-job> --reason TEXT   no new leases on the node, current holder and queued
                                                 pinned waiters finish, the tick releases it when idle,
                                                 the host joins the permanent exclude list
  broker.py renew --target N | --per-arch b200=3,gb200=1 [--wait 0]
                                                 re-request when the pool would drop below the target
                                                 within 45 min, in total or per arch
  broker.py tick --target N | --per-arch SPEC [--wait 10m]
                                                 reconcile, release unheld nodes with < 5 min left,
                                                 renew, bootstrap every requested node that is running
  broker.py fetch <holder>                       tar <root>/results/<holder>/ from every ready node
                                                 into state/results/<holder>/

tools/broker_loop.sh runs tick periodically. Every ssh call uses BatchMode, ConnectTimeout 20 and
StrictHostKeyChecking=accept-new. Remote output is kept short, everything is logged to
state/logs/broker.log and state/logs/bootstrap_<node>.log. Exit codes: 0 ok, 1 error, 3 cap refused.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swarmlib import (ACTIVE_STATES, LOGS, RESULTS, SWARM, Registry, fmt_remaining, iso,  # noqa: E402
                      node_label, parse_duration, parse_iso, remaining_seconds, utcnow)

USER = "estehle"
HOLD_SCRIPT = "~/b200_alloc/topk_hold.sh"
JOB_NAME = "topk-swarm"
CLUSTERS = {
    "computelab": dict(
        login="computelab-sc-01", arch="b200",
        sbatch='sbatch --job-name=%s{exclude} -p "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,'
               'b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb" --gres=gpu:b200:1 -t 4:00:00 %s'
               % (JOB_NAME, HOLD_SCRIPT),
        bootstrap="bootstrap_computelab.sh",
        image="nvidia/cuda:13.4.2-devel-ubuntu24.04"),
    "dlcluster": dict(
        login="dlcluster", arch="gb200",
        sbatch='sbatch --job-name=%s{exclude} -p gb200nvl4,gb200nvl72_preprod -A blackwell --gres=gpu:GB200:1 '
               '-t 4:00:00 %s' % (JOB_NAME, HOLD_SCRIPT),
        bootstrap="bootstrap_dlcluster.sh",
        image="nvidia/cuda:13.4.2-devel-ubuntu24.04"),
}
ARCH_TO_CLUSTER = {c["arch"]: name for name, c in CLUSTERS.items()}
# Hosts never requested again in this study (plus whatever `retire` adds to registry["exclude_hosts"]).
# umbriel-b200-020: deterministic slow outlier for kernels above about 64 KB of SASS, 7 to 51 percent
# slower than 047, 068 and 091 on the same binaries (lead's decision 2026-10-25 after verify-N8 and
# hw-researcher probes). Not a final evaluation node, no further cards.
EXCLUDE_HOSTS = {"computelab": ["umbriel-b200-020"], "dlcluster": []}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new"]


def parse_per_arch(spec):
    """'b200=3,gb200=1' -> {'b200': 3, 'gb200': 1}"""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        arch, _, n = part.partition("=")
        if arch not in ARCH_TO_CLUSTER or not n.isdigit():
            raise ValueError("bad per-arch target %r, expected e.g. b200=3,gb200=1" % spec)
        out[arch] = int(n)
    if not out:
        raise ValueError("empty per-arch target")
    return out
EXPIRING_S = 45 * 60      # below this a ready node becomes expiring and counts as gone for renew
RELEASE_S = 5 * 60        # below this an unheld node is released (cleanup) instead of dying on its own
BROKER_LOG = LOGS / "broker.log"
SQUEUE_FMT = "%i|%j|%T|%L|%N|%P|%l|%S"


# ----------------------------------------------------------------------------- helpers
def log(msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(BROKER_LOG, "a") as f:
        f.write("%s %s\n" % (iso(utcnow()), msg))


def _short(s, n=300):
    return s[:n].replace("\n", "|")


class Result:
    def __init__(self, rc, out, err):
        self.rc, self.out, self.err = rc, out, err

    @property
    def ok(self):
        return self.rc == 0


def ssh(host, cmd, timeout=180, jump=None, binary=False):
    """Run cmd on host. Returns Result (rc=-1 on timeout or spawn failure)."""
    argv = ["ssh"] + SSH_OPTS
    if jump:
        argv += ["-J", jump]
    argv += [host, cmd]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("ssh %s%s TIMEOUT(%ds) cmd=%s" % (host, " via " + jump if jump else "", timeout, _short(cmd, 200)))
        return Result(-1, b"" if binary else "", "timeout after %ds" % timeout)
    except OSError as e:
        log("ssh %s spawn error %s" % (host, e))
        return Result(-1, b"" if binary else "", str(e))
    out = p.stdout if binary else p.stdout.decode(errors="replace")
    err = p.stderr.decode(errors="replace")
    log("ssh %s%s rc=%d cmd=%s out=%s err=%s" % (
        host, " via " + jump if jump else "", p.returncode, _short(cmd, 200),
        ("<%d bytes>" % len(out)) if binary else _short(out), _short(err)))
    return Result(p.returncode, out, err)


def scp(src, host, dst, jump=None, timeout=120):
    argv = ["scp", "-q"] + SSH_OPTS
    if jump:
        argv += ["-J", jump]
    argv += [str(src), "%s:%s" % (host, dst)]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log("scp %s -> %s:%s TIMEOUT" % (src, host, dst))
        return False
    log("scp %s -> %s:%s rc=%d err=%s" % (src, host, dst, p.returncode, _short(p.stderr.decode(errors="replace"))))
    return p.returncode == 0


def slurm_time_to_seconds(s):
    """D-HH:MM:SS, HH:MM:SS, MM:SS or SS. None for INVALID, UNLIMITED, N/A."""
    s = (s or "").strip()
    if not s or not s[0].isdigit():
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def squeue(cluster):
    """Returns {job_id: {...}} for the user's jobs on the cluster, or None if the login host failed."""
    login = CLUSTERS[cluster]["login"]
    r = ssh(login, 'squeue -u %s -h -o "%s"' % (USER, SQUEUE_FMT), timeout=60)
    if not r.ok:
        return None
    jobs = {}
    for line in r.out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 7:
            continue
        jobs[parts[0]] = dict(name=parts[1], state=parts[2], left=parts[3], nodes=parts[4],
                              partition=parts[5], limit=parts[6], start=parts[7] if len(parts) > 7 else "")
    return jobs


def all_queues():
    return {c: squeue(c) for c in CLUSTERS}


def audit_holders(node, event, reason):
    """Record holder-clearing broker actions in state/logs/lease.log next to the lease tool's own lines."""
    holders = [g["holder"] for g in node.get("gpus", []) if g.get("holder")]
    if not holders:
        return
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LOGS / "lease.log", "a") as f:
            f.write("%s broker-%s holder=%s pid=%d node=%s %s\n" % (
                iso(utcnow()), event, ",".join(holders), os.getpid(), node_label(node), reason))
    except OSError:
        pass


def mark_lost(node, reason):
    audit_holders(node, "lost", reason)
    node["status"] = "lost"
    node["lost_reason"] = reason
    for g in node.get("gpus", []):
        g["holder"] = None
        g["since"] = None


def reconcile(reg, queues, now=None):
    """Update active node entries from squeue output. Returns a list of notes."""
    now = now or utcnow()
    notes = []
    for n in reg["nodes"]:
        if n.get("status") not in ACTIVE_STATES:
            continue
        jobs = queues.get(n["cluster"])
        if jobs is None:
            notes.append("%s: login host %s unreachable, left as %s" % (node_label(n), CLUSTERS[n["cluster"]]["login"], n["status"]))
            continue
        job = jobs.get(str(n["job_id"]))
        if job is None:
            notes.append("%s: job %s no longer in squeue, marked lost" % (node_label(n), n["job_id"]))
            mark_lost(n, "vanished from squeue at %s" % iso(now))
            continue
        if job["state"] == "RUNNING":
            left = slurm_time_to_seconds(job["left"])
            limit = slurm_time_to_seconds(job["limit"])
            if left is not None:
                n["lease_expires"] = iso(now + datetime.timedelta(seconds=left))
                if limit and not n.get("lease_start"):
                    n["lease_start"] = iso(now + datetime.timedelta(seconds=left - limit))
            if job["nodes"] and (not n.get("ssh") or n["ssh"] == "pending"):
                host = job["nodes"]
                n["ssh"] = host
                taken = any(o is not n and o.get("status") in ACTIVE_STATES and o.get("name") == host for o in reg["nodes"])
                n["name"] = host if not taken else "%s-j%s" % (host, n["job_id"])
            n["partition"] = job["partition"]
            if left is not None and left <= 0:
                notes.append("%s: lease is over, marked lost" % node_label(n))
                mark_lost(n, "time limit reached")
            elif n["status"] == "ready" and left is not None and left < EXPIRING_S:
                n["status"] = "expiring"
                notes.append("%s: %s left, now expiring" % (node_label(n), fmt_remaining(left)))
        elif job["state"] in ("PENDING", "CONFIGURING"):
            n["pending_reason"] = job.get("nodes") or ""
        else:
            notes.append("%s: job %s is %s, marked lost" % (node_label(n), n["job_id"], job["state"]))
            mark_lost(n, "slurm state %s at %s" % (job["state"], iso(now)))
    known = {str(n["job_id"]) for n in reg["nodes"]}
    for c, jobs in queues.items():
        for jid, j in (jobs or {}).items():
            if j["name"] == JOB_NAME and jid not in known and j["state"] in ("RUNNING", "PENDING"):
                notes.append("%s: unregistered %s job %s (%s on %s), register or scancel by hand"
                             % (c, JOB_NAME, jid, j["state"], j["nodes"] or "-"))
    return notes


def count_active(reg):
    return sum(1 for n in reg["nodes"] if n.get("status") in ACTIVE_STATES)


def find_node(reg, key):
    key = str(key)
    for want_active in (True, False):
        for n in reg["nodes"]:
            if (n.get("status") in ACTIVE_STATES) != want_active:
                continue
            if key in (n.get("name"), n.get("ssh"), str(n.get("job_id"))):
                return n
    return None


def good_nodes(reg, now=None):
    """Nodes that will still be usable in 45 minutes: requested, or ready/bootstrapping with time left."""
    now = now or utcnow()
    good = []
    for n in reg["nodes"]:
        if n.get("status") == "requested":
            good.append(n)
        elif n.get("status") in ("ready", "bootstrapping"):
            rem = remaining_seconds(n, now)
            if rem is None or rem > EXPIRING_S:
                good.append(n)
    return good


def print_table(reg, show_all=False):
    now = utcnow()
    fmt = "%-22s %-10s %-5s %-13s %-8s %-9s %-7s %s"
    print(fmt % ("node", "cluster", "arch", "status", "job", "remaining", "cuda", "gpus"))
    for n in reg["nodes"]:
        if n.get("status") in ("released", "lost") and not show_all:
            continue
        gpus = ", ".join("gpu%s=%s" % (g.get("index"), g.get("holder") or "free") for g in n.get("gpus", [])) or "-"
        if n.get("retire"):
            gpus += "  [retiring]"
        print(fmt % (node_label(n), n.get("cluster"), n.get("arch"), n.get("status"), n.get("job_id"),
                     fmt_remaining(remaining_seconds(n, now)), n.get("cuda") or "-", gpus))
    print("active=%d cap=%d good_for_45min=%d" % (count_active(reg), reg.get("cap", 6), len(good_nodes(reg, now))))


def submit(reg, cluster, nodelist=None, note=None):
    """sbatch a hold on cluster and append a `requested` entry. Caller holds the registry lock.
    Hosts that already carry an active entry on this cluster and the permanent exclude list are
    excluded (one job per host). `nodelist` pins the request to one host and bypasses the exclude
    list for an explicit one-off, which is recorded in the entry."""
    c = CLUSTERS[cluster]
    busy = {n["ssh"] for n in reg["nodes"]
            if n.get("cluster") == cluster and n.get("status") in ACTIVE_STATES and n.get("ssh")}
    banned = set(EXCLUDE_HOSTS.get(cluster, [])) | set(reg.get("exclude_hosts", {}).get(cluster, []))
    busy = sorted(busy | banned)
    if nodelist:
        exclude = " --nodelist=%s" % nodelist
    else:
        exclude = (" --exclude=" + ",".join(busy)) if busy else ""
    r = ssh(c["login"], c["sbatch"].format(exclude=exclude), timeout=90)
    m = re.search(r"Submitted batch job (\d+)", r.out or "")
    if not m:
        print("request %s failed: %s %s" % (cluster, _short(r.out, 200), _short(r.err, 200)))
        return None
    jid = m.group(1)
    entry = {
        "name": "pending-%s" % jid, "cluster": cluster, "arch": c["arch"], "login": c["login"],
        "ssh": None, "job_id": jid, "lease_start": None, "lease_expires": None, "status": "requested",
        "root": None, "exec_prefix": None, "container": None, "cuda": None, "nvcc": None, "cmake": None,
        "clone_commit": None, "gpus": [], "requested_at": iso(utcnow()), "excluded_hosts": [] if nodelist else busy,
    }
    if nodelist:
        entry["nodelist"] = nodelist
        entry["one_off"] = note or "pinned request"
    reg["nodes"].append(entry)
    print("requested %s job %s (%s)%s" % (cluster, jid, c["arch"],
                                         (" pinned to " + nodelist) if nodelist else ((" excluding " + ",".join(busy)) if busy else "")))
    return jid


def wait_running(job_ids, wait_s, poll=30):
    """Poll squeue until every job in job_ids is RUNNING or wait_s elapses. Returns the set still not running."""
    pending = set(str(j) for j in job_ids)
    deadline = time.monotonic() + wait_s
    while pending:
        queues = all_queues()
        with Registry() as reg:
            reconcile(reg.data, queues)
            reg.save()
            for n in reg.data["nodes"]:
                if str(n["job_id"]) in pending and n.get("ssh") and n["status"] in ACTIVE_STATES:
                    print("job %s running on %s (%s left)" % (n["job_id"], n["ssh"], fmt_remaining(remaining_seconds(n))))
                    pending.discard(str(n["job_id"]))
                elif str(n["job_id"]) in pending and n["status"] not in ACTIVE_STATES:
                    print("job %s is %s" % (n["job_id"], n["status"]))
                    pending.discard(str(n["job_id"]))
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(min(poll, max(1, deadline - time.monotonic())))
    return pending


def gpu_identity(login, jid):
    """Return (uuid, name, driver) of the job's GPU, resolved inside the job with srun --overlap."""
    cmd = ("srun --jobid=%s --overlap -N1 -n1 bash -c 'echo CVD=$CUDA_VISIBLE_DEVICES; "
           "nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader'" % jid)
    r = ssh(login, cmd, timeout=150)
    if not r.ok:
        return None
    cvd = None
    rows = []
    for line in r.out.splitlines():
        line = line.strip()
        if line.startswith("CVD="):
            v = line[4:].split(",")[0].strip()
            cvd = int(v) if v.isdigit() else None
        elif re.match(r"^\d+,\s*GPU-", line):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                rows.append(parts)
    if not rows:
        return None
    pick = None
    if cvd is not None:
        for p in rows:
            if p[0].isdigit() and int(p[0]) == cvd:
                pick = p
    if pick is None:
        pick = rows[0]
    return pick[1], pick[2], pick[3]


def node_cleanup(node):
    """Best effort: remove the docker container so nothing references the GPU after the lease."""
    if node.get("cluster") == "computelab" and node.get("container") and node.get("ssh"):
        r = ssh(node["ssh"], "docker rm -f %s >/dev/null 2>&1 && echo removed || echo none" % node["container"],
                timeout=60, jump=node.get("jump"))
        return r.out.strip() if r.rc != -1 else "unreachable"
    return "nothing to clean"


def release_node(key, force=False):
    """Cleanup, scancel and mark released. Returns 0 ok, 1 refused or error."""
    with Registry() as reg:
        node = find_node(reg.data, key)
        if node is None:
            print("no such node or job: %s" % key)
            return 1
        holders = [g["holder"] for g in node.get("gpus", []) if g.get("holder")]
        if holders and not force:
            print("%s: still held by %s, release their leases first or pass --force" % (node_label(node), ", ".join(holders)))
            return 1
        snapshot = dict(node)
    cleaned = node_cleanup(snapshot) if snapshot.get("status") in ("bootstrapping", "ready", "expiring") else "skipped"
    r = ssh(snapshot["login"], "scancel %s" % snapshot["job_id"], timeout=60)
    with Registry() as reg:
        node = find_node(reg.data, snapshot["job_id"])
        if node is not None:
            audit_holders(node, "released", "force=%s" % force)
            node["status"] = "released"
            node["released_at"] = iso(utcnow())
            for g in node.get("gpus", []):
                g["holder"] = None
                g["since"] = None
            reg.save()
    print("%s: container %s, scancel %s rc=%d, marked released" % (node_label(snapshot), cleaned, snapshot["job_id"], r.rc))
    return 0 if r.ok else 1


def retire_node(key, reason):
    """Mark a node retiring: lease.py gives no new leases (queued pinned waiters and the current holder
    may finish), the tick releases it once idle, and the host joins registry["exclude_hosts"]."""
    with Registry() as reg:
        node = find_node(reg.data, key)
        if node is None:
            print("no such node or job: %s" % key)
            return 1
        node["retire"] = reason
        node["retire_at"] = iso(utcnow())
        ex = reg.data.setdefault("exclude_hosts", {})
        lst = ex.setdefault(node["cluster"], [])
        if node.get("ssh") and node["ssh"] not in lst:
            lst.append(node["ssh"])
        reg.save()
        holders = [g["holder"] for g in node.get("gpus", []) if g.get("holder")]
        queued = [w["holder"] for g in node.get("gpus", []) for w in (g.get("queue") or [])]
        print("%s: retiring (%s). holders=%s queued=%s, excluded on %s from now on" % (
            node_label(node), reason, ",".join(holders) or "none", ",".join(queued) or "none", node["cluster"]))
    return 0


def bootstrap_node(key, image=None, timeout=2400, force=False):
    """Run the on-node bootstrap for a running job and mark it ready. Returns 0 ok, 1 error, 2 duplicate host.
    A node already in `bootstrapping` for less than 20 minutes is skipped unless force, so the tick loop
    and a hand-run bootstrap never work on the same job at once."""
    queues = all_queues()
    with Registry() as reg:
        reconcile(reg.data, queues)
        node = find_node(reg.data, key)
        if node is None:
            print("no such node or job: %s" % key)
            reg.save()
            return 1
        if node["status"] not in ACTIVE_STATES:
            print("%s is %s, not bootstrapping" % (node_label(node), node["status"]))
            reg.save()
            return 1
        started = parse_iso(node.get("bootstrap_started"))
        if node["status"] == "bootstrapping" and not force and started is not None \
                and (utcnow() - started).total_seconds() < 20 * 60:
            print("%s: bootstrap already running since %s, skipping (pass --force to retry)" % (node_label(node), node["bootstrap_started"]))
            reg.save()
            return 1
        if not node.get("ssh"):
            print("job %s is not running yet (%s)" % (node["job_id"], node.get("pending_reason") or "pending"))
            reg.save()
            return 1
        dup = [o for o in reg.data["nodes"] if o is not node and o.get("ssh") == node["ssh"]
               and o.get("status") in ("bootstrapping", "ready", "expiring")]
        if dup:
            print("%s: host already in use by job %s, releasing this duplicate job %s"
                  % (node["ssh"], dup[0]["job_id"], node["job_id"]))
            reg.save()
            release_node(node["job_id"], force=True)
            return 2
        node["status"] = "bootstrapping"
        node["bootstrap_started"] = iso(utcnow())
        reg.save()
        cluster, login, host, jid = node["cluster"], node["login"], node["ssh"], node["job_id"]
        label = node_label(node)
    c = CLUSTERS[cluster]
    image = image or c["image"]

    ident = gpu_identity(login, jid)
    if ident is None:
        print("%s: could not resolve the job's GPU with srun --overlap, see %s" % (label, BROKER_LOG))
        return 1
    uuid, gpu_name, driver = ident
    print("%s: job gpu %s (%s, driver %s)" % (label, uuid, gpu_name, driver))

    jump = None
    if not ssh(host, "hostname", timeout=40).ok:
        jump = login
        if not ssh(host, "hostname", timeout=40, jump=jump).ok:
            print("%s: cannot ssh to the node directly or via %s" % (label, login))
            return 1
        print("%s: direct ssh failed, using -J %s" % (label, login))

    script = SWARM / "nodes" / c["bootstrap"]
    remote = "/tmp/%s_%s" % (USER, script.name)
    if not scp(script, host, remote, jump=jump):
        print("%s: scp of %s failed" % (label, script.name))
        return 1
    r = ssh(host, "bash %s %s %s" % (remote, shlex.quote(uuid), shlex.quote(image)), timeout=timeout, jump=jump)
    blog = LOGS / ("bootstrap_%s.log" % label)
    with open(blog, "a") as f:
        f.write("=== %s bootstrap %s job %s rc=%d\n%s\n--- stderr\n%s\n" % (iso(utcnow()), label, jid, r.rc, r.out, r.err))
    tail = (r.out or "").strip().splitlines()
    for line in tail[-8:]:
        if not line.startswith("BOOTSTRAP_JSON"):
            print("  " + line[:160])
    m = re.search(r"^BOOTSTRAP_JSON (.*)$", r.out or "", re.M)
    if not r.ok or not m:
        print("%s: bootstrap failed (rc=%d), node left in bootstrapping, log %s" % (label, r.rc, blog))
        if r.err.strip():
            print("  stderr: " + _short(r.err, 300))
        return 1
    info = json.loads(m.group(1))
    with Registry() as reg:
        node = find_node(reg.data, jid)
        for k in ("root", "host_root", "exec_prefix", "container", "image", "cuda", "nvcc", "nvcc_version",
                  "cmake", "cmake_version", "clone_commit", "driver", "gpu_name", "gpu_uuid", "concurrent_exec"):
            if k in info:
                node[k] = info[k]
        node["jump"] = jump
        node["bootstrapped_at"] = iso(utcnow())
        old = {g.get("index"): g for g in node.get("gpus", [])}
        idx = info.get("gpu_index", 0)
        keep = old.get(idx, {})
        node["gpus"] = [{"index": idx, "holder": keep.get("holder"), "since": keep.get("since")}]
        rem = remaining_seconds(node)
        node["status"] = "expiring" if (rem is not None and rem < EXPIRING_S) else "ready"
        reg.save()
        print("%s: %s  cuda=%s nvcc=%s cmake=%s driver=%s commit=%s root=%s exec_prefix=%r expires=%s" % (
            label, node["status"], node.get("cuda"), node.get("nvcc_version"), node.get("cmake_version"),
            node.get("driver"), node.get("clone_commit"), node.get("root"), node.get("exec_prefix"),
            node.get("lease_expires")))
    return 0


def renew_pool(target=None, wait_s=0, queues=None, per_arch=None):
    """Request holds until the pool is good for the next 45 minutes: `target` nodes in total (at least one
    per arch, then alternating clusters), or `per_arch` = {'b200': 3, 'gb200': 1}. Returns (rc, new job ids)."""
    queues = queues if queues is not None else all_queues()
    jids = []
    with Registry() as reg:
        notes = reconcile(reg.data, queues)
        good = good_nodes(reg.data)
        cap = reg.data.get("cap", 6)
        room = cap - count_active(reg.data)
        reg.save()
        if per_arch:
            counts = {a: sum(1 for n in good if n.get("arch") == a) for a in per_arch}
            deficits = {a: max(0, per_arch[a] - counts[a]) for a in per_arch}
            need = sum(deficits.values())
            desc = ", ".join("%s %d/%d" % (a, counts[a], per_arch[a]) for a in sorted(per_arch))
        else:
            need = target - len(good)
            desc = "%d/%d" % (len(good), target)
        if need <= 0:
            print("pool ok: good for the next 45 min %s" % desc)
            rc = 0
        elif room <= 0:
            print("pool below target (%s) but cap %d is full, waiting for expiries or releases" % (desc, cap))
            rc = 3
        else:
            need = min(need, room)
            reachable = [c for c in CLUSTERS if queues.get(c) is not None]
            if not reachable:
                print("no login host reachable, nothing requested")
                rc = 1
            else:
                order = []
                if per_arch:
                    d = dict(deficits)
                    while sum(d.values()) > 0:
                        for a in sorted(d, key=lambda a: -d[a]):
                            if d[a] > 0:
                                d[a] -= 1
                                if ARCH_TO_CLUSTER[a] in reachable:
                                    order.append(ARCH_TO_CLUSTER[a])
                else:
                    have_arch = {n["arch"] for n in good}
                    order = [c for c in reachable if CLUSTERS[c]["arch"] not in have_arch]
                    while len(order) < need:
                        order.extend(reachable)
                if not order:
                    print("pool below target (%s) but the needed cluster is unreachable" % desc)
                for cluster in order[:need]:
                    jid = submit(reg.data, cluster)
                    if jid:
                        jids.append(jid)
                reg.save()
                rc = 0 if jids else 1
    for note in notes:
        print("note: " + note)
    if jids and wait_s > 0:
        left = wait_running(jids, wait_s)
        if left:
            print("still pending after %ds: %s" % (wait_s, ", ".join(sorted(left))))
    return rc, jids


# ----------------------------------------------------------------------------- commands
def cmd_status(args):
    queues = all_queues()
    with Registry() as reg:
        notes = reconcile(reg.data, queues)
        reg.save()
        data = reg.data
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_table(data, show_all=args.all)
    for c, q in queues.items():
        if q is None:
            print("note: %s login host unreachable" % c)
    for note in notes:
        print("note: " + note)
    return 0


def cmd_request(args):
    jids = []
    with Registry() as reg:
        active = count_active(reg.data)
        cap = reg.data.get("cap", 6)
        if active + args.count > cap:
            print("refused: %d active + %d requested would exceed cap %d" % (active, args.count, cap))
            return 3
        for _ in range(args.count):
            jid = submit(reg.data, args.cluster, nodelist=args.nodelist, note=args.note)
            if jid:
                jids.append(jid)
        reg.save()
    if not jids:
        return 1
    wait_s = parse_duration(args.wait)
    if wait_s > 0:
        left = wait_running(jids, wait_s)
        if left:
            print("still pending after %s: %s" % (args.wait, ", ".join(sorted(left))))
    return 0


def cmd_bootstrap(args):
    return bootstrap_node(args.node, image=args.image, timeout=args.timeout, force=args.force)


def cmd_release(args):
    return release_node(args.node, force=args.force)


def cmd_retire(args):
    return retire_node(args.node, args.reason)


def target_args(args):
    per_arch = parse_per_arch(args.per_arch) if args.per_arch else None
    if per_arch is None and args.target is None:
        raise SystemExit("one of --target N or --per-arch b200=3,gb200=1 is required")
    return args.target, per_arch


def cmd_renew(args):
    target, per_arch = target_args(args)
    rc, _ = renew_pool(target, parse_duration(args.wait), per_arch=per_arch)
    return rc


def cmd_tick(args):
    """One maintenance pass. Safe to run every few minutes."""
    target, per_arch = target_args(args)
    queues = all_queues()
    with Registry() as reg:
        notes = reconcile(reg.data, queues)
        reg.save()
        now = utcnow()
        dying = []
        for n in reg.data["nodes"]:
            if n.get("status") not in ("ready", "expiring", "bootstrapping"):
                continue
            rem = remaining_seconds(n, now)
            held = any(g.get("holder") for g in n.get("gpus", []))
            queued = any((g.get("queue") or []) for g in n.get("gpus", []))
            if n.get("retire") and not held and not queued:
                notes.append("%s: retired and idle, releasing" % node_label(n))
                dying.append(n["job_id"])
            elif rem is not None and rem < RELEASE_S and not held:
                dying.append(n["job_id"])
            elif rem is not None and rem < RELEASE_S and held:
                notes.append("%s: %s left and still held by %s, slurm will kill it" % (
                    node_label(n), fmt_remaining(rem), ", ".join(g["holder"] for g in n["gpus"] if g.get("holder"))))
    for note in notes:
        print("note: " + note)
    for jid in dying:
        release_node(jid)
    rc, _ = renew_pool(target, parse_duration(args.wait), queues=None, per_arch=per_arch)
    if rc == 3:
        # The cap blocks a renewal. Give up unheld expiring nodes now instead of at the 5 minute mark,
        # so the replacement still gets most of the 45 minute overlap.
        with Registry() as reg:
            idle_expiring = [n["job_id"] for n in reg.data["nodes"]
                             if n.get("status") == "expiring" and not any(g.get("holder") for g in n.get("gpus", []))]
        for jid in idle_expiring:
            print("cap full and renewal pending, releasing idle expiring job %s" % jid)
            release_node(jid)
        if idle_expiring:
            rc, _ = renew_pool(target, parse_duration(args.wait), queues=None, per_arch=per_arch)
    with Registry() as reg:
        todo = [n["job_id"] for n in reg.data["nodes"] if n.get("status") == "requested" and n.get("ssh")]
    for jid in todo:
        bootstrap_node(jid, image=args.image)
    with Registry() as reg:
        print_table(reg.data)
    return rc


def cmd_fetch(args):
    holder = args.holder
    with Registry() as reg:
        nodes = [dict(n) for n in reg.data["nodes"] if n.get("status") in ("ready", "expiring")]
    if not nodes:
        print("no ready nodes")
        return 1
    dest = RESULTS / holder
    archives = RESULTS / "_archives" / holder
    dest.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    total = 0
    for n in nodes:
        root = n["root"]
        inner = "test -d %s/results/%s && tar czf - -C %s/results %s" % (root, holder, root, holder)
        cmd = ("%s bash -c %s" % (n["exec_prefix"], shlex.quote(inner))) if n.get("exec_prefix") else ("bash -c %s" % shlex.quote(inner))
        r = ssh(n["ssh"], cmd, timeout=args.timeout, jump=n.get("jump"), binary=True)
        if r.rc == -1:
            print("%s: ssh failed (%s)" % (node_label(n), r.err.strip()))
            continue
        if not r.ok or not r.out:
            print("%s: no results for %s" % (node_label(n), holder))
            continue
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        tgz = archives / ("%s_%s.tgz" % (node_label(n), stamp))
        tgz.write_bytes(r.out)
        count = 0
        try:
            with tarfile.open(tgz) as t:
                members = []
                for m in t.getmembers():
                    parts = Path(m.name).parts
                    if len(parts) < 2 or parts[0] != holder or ".." in parts or m.name.startswith("/"):
                        continue
                    if not (m.isfile() or m.isdir()):
                        continue
                    m.name = str(Path(*parts[1:]))
                    members.append(m)
                    if m.isfile():
                        count += 1
                t.extractall(dest, members=members)
        except tarfile.TarError as e:
            print("%s: bad archive %s (%s)" % (node_label(n), tgz, e))
            continue
        total += count
        print("%s: %d file(s) -> %s (archive %s)" % (node_label(n), count, dest, tgz.name))
    print("fetched %d file(s) for %s" % (total, holder))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s.add_argument("--all", action="store_true", help="include released and lost nodes")
    s.set_defaults(func=cmd_status)
    r = sub.add_parser("request")
    r.add_argument("--cluster", required=True, choices=sorted(CLUSTERS))
    r.add_argument("--count", type=int, default=1)
    r.add_argument("--wait", default="0", help="wait this long for the job(s) to start, e.g. 30m")
    r.add_argument("--nodelist", default=None, help="pin to one host (bypasses the exclude list, one-off)")
    r.add_argument("--note", default=None, help="why this one-off exists, stored in the entry")
    r.set_defaults(func=cmd_request)
    b = sub.add_parser("bootstrap")
    b.add_argument("node", help="node name, ssh host or job id")
    b.add_argument("--image", default=None)
    b.add_argument("--timeout", type=int, default=2400, help="seconds for the on-node bootstrap script")
    b.add_argument("--force", action="store_true", help="retry a node that is already marked bootstrapping")
    b.set_defaults(func=cmd_bootstrap)
    rl = sub.add_parser("release")
    rl.add_argument("node")
    rl.add_argument("--force", action="store_true", help="release even if a holder still has a lease")
    rl.set_defaults(func=cmd_release)
    rt = sub.add_parser("retire")
    rt.add_argument("node")
    rt.add_argument("--reason", required=True)
    rt.set_defaults(func=cmd_retire)
    rn = sub.add_parser("renew")
    rn.add_argument("--target", type=int, default=None, help="total nodes good for the next 45 min")
    rn.add_argument("--per-arch", default=None, help="per-arch target, e.g. b200=3,gb200=1 (overrides --target)")
    rn.add_argument("--wait", default="0")
    rn.set_defaults(func=cmd_renew)
    t = sub.add_parser("tick")
    t.add_argument("--target", type=int, default=None, help="total nodes good for the next 45 min")
    t.add_argument("--per-arch", default=None, help="per-arch target, e.g. b200=3,gb200=1 (overrides --target)")
    t.add_argument("--wait", default="10m", help="how long to wait for newly requested jobs to start")
    t.add_argument("--image", default=None)
    t.set_defaults(func=cmd_tick)
    f = sub.add_parser("fetch")
    f.add_argument("holder")
    f.add_argument("--timeout", type=int, default=600)
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
