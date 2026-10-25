# Node registry protocol

The broker owns the pool. Workers only take and return GPU leases. Everything is file based under `$SWARM/state/` (gitignored) so it survives agent restarts and context compaction.

## Registry `state/nodes.json`

```
{
  "cap": 6,
  "updated": "2026-10-25T10:00:00Z",
  "nodes": [
    {
      "name": "umbriel-b200-018",
      "cluster": "computelab",            // computelab | dlcluster
      "arch": "b200",                      // b200 | gb200
      "login": "computelab-sc-01",         // ssh login host used to submit and attach
      "ssh": "umbriel-b200-018",           // what workers ssh to (direct, or via the login host if direct fails)
      "job_id": "5150030",
      "lease_start": "2026-10-25T09:40:00Z",
      "lease_expires": "2026-10-25T13:40:00Z",
      "status": "ready",                   // requested | bootstrapping | ready | expiring | released | lost
      "root": "/work/topk_swarm",          // on-node root, see layout below
      "exec_prefix": "docker exec -i topk_swarm",   // how to run a command inside the container, empty if bare
      "container": "topk_swarm",
      "cuda": "13.4.2",
      "nvcc": "/usr/local/cuda/bin/nvcc",
      "cmake": "/opt/cmake/bin/cmake",
      "clone_commit": "639effa277",
      "gpus": [ {"index": 0, "holder": null, "since": null} ]
    }
  ]
}
```

Only the broker adds, changes or removes node entries. `tools/lease.py` changes only `gpus[].holder` and `gpus[].since`, under `fcntl.flock` on `state/nodes.lock`, and writes the file atomically (write to a temp file, then rename). Workers never edit the file by hand.

## Lease tool `tools/lease.py`

```
python3 tools/lease.py acquire --holder <name> --arch any|b200|gb200 [--min-remaining 45m] [--wait 30m]
python3 tools/lease.py release --holder <name>
python3 tools/lease.py status
```

`acquire` picks a ready node with a free GPU and at least the requested remaining lease, marks it held, prints one line `node=<ssh> gpu=<index> root=<root> exec_prefix=<prefix> arch=<arch> cuda=<cuda> expires=<iso>` and exits 0. With nothing free it waits up to `--wait` polling the file, then exits 2. `release` frees every GPU held by that holder. A holder keeps a lease only while a batch runs. The broker may revoke a lease on a node in `expiring` state after warning the holder by message.

## On-node layout under `root`

```
<root>/cccl            blobless clone of the fork, branch exp/topk-primitives-swarm
<root>/locks/gpu<i>.lock   one lock file per GPU, every measurement runs under flock on it
<root>/work/<holder>/  per-agent scratch, builds, worktrees
<root>/results/<holder>/   JSON results, fetched to the Mac by the holder
<root>/logs/           broker and bootstrap logs
```

Measurement rule: `CUDA_VISIBLE_DEVICES=<i> flock <root>/locks/gpu<i>.lock -c '<variant and control in one command>'`. Compiles run outside the lock.

## Allocation

Computelab (B200, x86): login `ssh computelab-sc-01`. Detachable hold: `sbatch --job-name=topk-swarm -p "<B200 partitions>" --gres=gpu:b200:1 -t 4:00:00 ~/b200_alloc/hold.sh`, attach with `srun --jobid=<id> --pty bash` or `ssh <node>`, release with `scancel <id>`. The crun equivalent is `crun -q 'cpu.arch=x86_64 and gpu.product_name="*B200*"' -i -w -t 4:00:00` and is the fallback when sbatch fails. Nodes have docker, the container is a CUDA devel image with `/work` on the local overlay. Details and pitfalls in `/Users/estehle/.claude/projects/-Users-estehle-Downloads-cccl/memory/computelab-b200-allocation.md`.

dlcluster (GB200, aarch64): login `ssh dlcluster`. Detachable hold: `sbatch -p gb200nvl4,gb200nvl72_preprod -A blackwell --gres=gpu:GB200:1 -t 4:00:00 ~/b200_alloc/hold.sh`, attach with `srun --jobid=<id> --pty bash -l`. The interactive form is `srun -p gb200nvl4 -A blackwell --gres=gpu:GB200:1 -t 4:00:00 --pty bash -l`. No docker daemon, no cmake on the bare node, pyxis and enroot work, local storage under `/raid`. Details in `/Users/estehle/.claude/projects/-Users-estehle-Downloads-cccl/memory/dlcluster-gb200-allocation.md`.

Cap: at most six nodes in `requested`, `bootstrapping`, `ready` or `expiring` state at any time across both clusters. Target pool two to four, at least one of each arch when both clusters grant. Release a node (`scancel`) when no holder has used it for 30 minutes and no wave is pending. Re-request ahead of expiry when the pool would drop below target. The 5 GB NFS home is not a work area, only `hold.sh` and small scripts live there.

## Broker notes (verified 2026-10-25, phase 0)

- `gpus[].index` is the CUDA device index as seen through `exec_prefix`. Both containers expose only the job's GPU, so the index is 0 on both arches and `CUDA_VISIBLE_DEVICES=0` with `locks/gpu0.lock` is always right. Never touch other GPUs of a node, they belong to other jobs.
- The lease line is `node=... gpu=... root=... exec_prefix='...' arch=... cuda=... expires=... login=...`. `exec_prefix` is shell quoted, parse the line with `shlex.split`. Direct ssh from the Mac to the node works on both clusters, the fallback is `ssh -J <login> <node>`.
- Running on a node from the Mac, compile outside the lock, measure inside, one ssh per step and short output:

  ```
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new <node> \
    "<exec_prefix> bash -c 'cd <root>/work/<holder> && nvcc -arch=sm_100 -O3 -o bench bench.cu > build.log 2>&1 && echo built'"
  ssh ... <node> "<exec_prefix> bash -c 'cd <root>/work/<holder> && CUDA_VISIBLE_DEVICES=0 flock <root>/locks/gpu0.lock -c \"./bench --variant x --control y\" > <root>/results/<holder>/run1.json && echo done'"
  python3 $SWARM/tools/broker.py fetch <holder>
  ```

- computelab B200: `exec_prefix` is `docker exec -i topk_swarm`, root inside the container, image `nvidia/cuda:13.4.2-devel-ubuntu24.04`, cmake 4.3.2 at `/opt/cmake/bin/cmake` and on `PATH`, host directory `/tmp/topk_swarm` is `/work`.
- dlcluster GB200: the bare node has no CUDA toolkit and `/raid` is not writable for the user, so the broker uses enroot directly, no pyxis, no slurm steps. `exec_prefix` is `ENROOT_RESTRICT_DEV=y enroot start --rw -e NVIDIA_VISIBLE_DEVICES=<uuid> -m /tmp/estehle/topk_swarm:/work topk_swarm`. Every command is a fresh container start (about half a second), concurrent starts work, the rootfs is persistent and writable, the NVIDIA entrypoint banner is removed. cmake 4.4.4 from a pip venv at `/opt/cmake/bin/cmake` and on `PATH`. For apt inside the container use `enroot start --root ...` and run `umount /etc/localtime` first, otherwise the tzdata postinst fails.
- Both containers have python3 3.12, git, ninja and flock. The clone is `<root>/cccl` on the swarm branch, update it with `git fetch` and `git reset --hard origin/<branch>`, never commit there.
- `tools/broker.py fetch <holder>` tars `<root>/results/<holder>/` from every ready node into `state/results/<holder>/` and keeps the archives under `state/results/_archives/<holder>/`.
- computelab slurm prints Pacific time, dlcluster prints UTC. The broker derives expiry from `squeue %L` (time left), so registry times are UTC on both.
