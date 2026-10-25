#!/usr/bin/env bash
# Bootstrap a dlcluster GB200 node (aarch64, no docker daemon, no CUDA toolkit on the bare node)
# for topk_swarm. Runs ON the node as the user.
#   bootstrap_dlcluster.sh <gpu-uuid> [docker-ref]
# The bare node has no nvcc, so this uses enroot directly (no slurm steps): imports the
# nvidia/cuda devel image once into <base>/images/*.sqsh, creates a persistent writable
# container named `topk_swarm`, binds <base>/topk_swarm at /work, installs git, python3, ninja
# and a venv cmake (aarch64 wheels), clones the fork branch blobless, and runs a trivial sm_100
# kernel under the gpu0 lock. Only the job's GPU is exposed (NVIDIA_VISIBLE_DEVICES=<uuid>, so it
# is CUDA index 0 inside). <base> is /raid/$USER when writable, else /tmp/$USER. Idempotent.
# Prints short progress lines and a final `BOOTSTRAP_JSON {...}` line for tools/broker.py.
set -u
UUID=${1:?gpu uuid}
REF=${2:-nvidia/cuda:13.4.2-devel-ubuntu24.04}
BRANCH=exp/topk-primitives-swarm
REPO=https://github.com/elstehle/cccl.git
CN=topk_swarm
ROOT=/work/topk_swarm            # root as seen through exec_prefix
if mkdir -p "/raid/$USER/topk_swarm" 2>/dev/null && [ -w "/raid/$USER/topk_swarm" ]; then BASE=/raid/$USER; else BASE=/tmp/$USER; fi
HOST_ROOT=$BASE/topk_swarm       # bind-mounted at /work inside the container
SQSH=$BASE/images/cuda-${REF##*/cuda:}.sqsh

mkdir -p "$HOST_ROOT/topk_swarm"/{locks,work,results,logs} "$BASE/images" || { echo "BOOTSTRAP_FAIL mkdir $HOST_ROOT"; exit 1; }
LOG=$HOST_ROOT/topk_swarm/logs/bootstrap.log
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { step "BOOTSTRAP_FAIL $*"; echo "--- log tail"; tail -n 12 "$LOG"; exit 1; }
# user-level exec prefix (what workers use) and the root-remapped one for apt.
# ENROOT_RESTRICT_DEV=y makes the 10-devices hook build a private /dev, so the nvidia hook
# exposes only NVIDIA_VISIBLE_DEVICES (our GPU, CUDA index 0) instead of the whole host /dev.
EXTRA_ENV=""
ex()  { ENROOT_RESTRICT_DEV=y enroot start --rw $EXTRA_ENV -e NVIDIA_VISIBLE_DEVICES="$UUID" -m "$HOST_ROOT":/work "$CN" "$@"; }
exr() { ENROOT_RESTRICT_DEV=y enroot start --root --rw $EXTRA_ENV -e NVIDIA_VISIBLE_DEVICES="$UUID" -m "$HOST_ROOT":/work "$CN" "$@"; }

step "bootstrap start host=$(hostname) arch=$(uname -m) uuid=$UUID image=$REF base=$BASE"
command -v enroot >/dev/null || fail "no enroot on host"
command -v flock >/dev/null || fail "no flock on host"

if ! enroot list 2>/dev/null | grep -qx "$CN"; then
  if [ ! -s "$SQSH" ]; then
    step "importing docker://$REF -> $SQSH"
    enroot import -o "$SQSH" "docker://$REF" >>"$LOG" 2>&1 || fail "enroot import"
  fi
  step "creating container $CN from $SQSH"
  enroot create -n "$CN" "$SQSH" >>"$LOG" 2>&1 || fail "enroot create"
else
  step "reusing container $CN"
fi

# The image's /etc/rc chains into /opt/nvidia/nvidia_entrypoint.sh, which prints the NGC license
# banner on every start. Replace it with a plain exec so command output stays clean.
exr bash -c 'printf "#!/bin/sh\ncd / 2>/dev/null\nif [ \$# -gt 0 ]; then exec \"\$@\"; else exec /bin/bash; fi\n" > /etc/rc' >>"$LOG" 2>&1 || fail "rewrite /etc/rc"
QUIET=$(ex bash -c 'echo quiet-ok' 2>>"$LOG") || fail "enroot start after rc rewrite"
[ "$QUIET" = quiet-ok ] || fail "enroot start output not clean: $QUIET"

# nvcc must be on PATH inside the container, otherwise pass the image PATH explicitly
if ! ex bash -c 'command -v nvcc' >/dev/null 2>>"$LOG"; then
  EXTRA_ENV="-e PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  ex bash -c 'command -v nvcc' >/dev/null 2>>"$LOG" || fail "nvcc not found in container even with explicit PATH"
  step "note: container needs explicit PATH, exec_prefix carries it"
fi

GPU_LINE=$(ex nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader 2>>"$LOG") || fail "nvidia-smi in container"
NGPU=$(printf '%s\n' "$GPU_LINE" | grep -c .)
[ "$NGPU" = 1 ] || fail "expected exactly 1 GPU in container, got $NGPU: $GPU_LINE"
printf '%s\n' "$GPU_LINE" | grep -q "$UUID" || fail "container GPU is not $UUID: $GPU_LINE"
step "container gpu: $GPU_LINE"

step "packages"
# enroot bind-mounts the host /etc/localtime (20-config.fstab), and tzdata's postinst cannot rename
# over a mount point, so unmount it in this start's private mount namespace before apt runs.
exr bash -c 'umount /etc/localtime 2>/dev/null; [ -e /etc/localtime ] || ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime; export DEBIAN_FRONTEND=noninteractive; (command -v git && command -v python3 && command -v ninja && command -v flock) >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 python3-venv python3-pip ninja-build util-linux ca-certificates less procps)' >>"$LOG" 2>&1 || fail "apt-get"
ex bash -c 'command -v flock' >/dev/null 2>&1 || fail "no flock in container"

step "cmake (venv pip, aarch64)"
ex bash -c 'test -x /work/venv/bin/cmake || (python3 -m venv /work/venv && /work/venv/bin/pip install -q cmake ninja)' >>"$LOG" 2>&1 || fail "pip cmake"
exr bash -c 'ln -sfn /work/venv /opt/cmake && ln -sf /work/venv/bin/cmake /usr/local/bin/cmake && ln -sf /work/venv/bin/ninja /usr/local/bin/ninja' >>"$LOG" 2>&1 || fail "cmake symlink"
CMAKE_VER=$(ex bash -c 'cmake --version | head -1 | grep -o "[0-9][0-9.a-z-]*"' 2>>"$LOG") || fail "cmake --version"

step "clone $BRANCH"
ex bash -c "test -d $ROOT/cccl/.git || git clone -q --filter=blob:none -b $BRANCH $REPO $ROOT/cccl" >>"$LOG" 2>&1 || fail "git clone"
ex bash -c "cd $ROOT/cccl && git fetch -q origin $BRANCH && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH" >>"$LOG" 2>&1 || fail "git update"
COMMIT=$(ex git -C $ROOT/cccl rev-parse --short=10 HEAD 2>>"$LOG") || fail "rev-parse"

step "smoke test (nvcc -arch=sm_100 -O3, under flock gpu0)"
cat > "$HOST_ROOT/topk_swarm/logs/smoke.cu" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int* out) { out[threadIdx.x] = threadIdx.x * 2; }
int main() {
  int* d = nullptr; int h[32] = {0};
  if (cudaMalloc(&d, sizeof(h)) != cudaSuccess) { printf("SMOKE_FAIL malloc %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  k<<<1, 32>>>(d);
  if (cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) { printf("SMOKE_FAIL memcpy %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  printf("SMOKE_OK gpu=\"%s\" sm=%d%d sms=%d rt=%d drv=%d last=%d\n", p.name, p.major, p.minor, p.multiProcessorCount, rt, drv, h[31]);
  return h[31] == 62 ? 0 : 1;
}
CU
ex bash -c "cd $ROOT/logs && nvcc -arch=sm_100 -O3 -o smoke smoke.cu" >>"$LOG" 2>&1 || fail "nvcc smoke.cu"
SMOKE=$(ex bash -c "CUDA_VISIBLE_DEVICES=0 flock $ROOT/locks/gpu0.lock -c $ROOT/logs/smoke" 2>>"$LOG") || fail "smoke run: $SMOKE"
echo "$SMOKE" >>"$LOG"
printf '%s\n' "$SMOKE" | grep -q SMOKE_OK || fail "smoke output: $SMOKE"
step "$SMOKE"

step "concurrency check (two enroot starts at once)"
( ex bash -c 'sleep 1; echo A' & ex bash -c 'sleep 1; echo B' & wait ) > "$HOST_ROOT/topk_swarm/logs/concurrency.txt" 2>>"$LOG"
CONC=$(sort "$HOST_ROOT/topk_swarm/logs/concurrency.txt" | tr -d '\n')
[ "$CONC" = AB ] && CONC_OK=yes || CONC_OK=no
step "concurrent enroot starts: $CONC_OK"

NVCC_VER=$(ex bash -c 'nvcc --version | grep -o "V[0-9][0-9.]*" | head -1')
CUDA_VER=$(ex bash -c "grep -o '\"version\" *: *\"[0-9.]*\"' /usr/local/cuda/version.json 2>/dev/null | head -1 | grep -o '[0-9][0-9.]*'")
[ -n "$CUDA_VER" ] || CUDA_VER=${NVCC_VER#V}
DRIVER=$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $4}')
GPU_NAME=$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $3}')
EXEC_PREFIX="ENROOT_RESTRICT_DEV=y enroot start --rw $EXTRA_ENV -e NVIDIA_VISIBLE_DEVICES=$UUID -m $HOST_ROOT:/work $CN"
EXEC_PREFIX=$(echo "$EXEC_PREFIX" | tr -s ' ')
step "done cuda=$CUDA_VER nvcc=$NVCC_VER cmake=$CMAKE_VER driver=$DRIVER commit=$COMMIT"
echo "BOOTSTRAP_JSON {\"root\": \"$ROOT\", \"host_root\": \"$HOST_ROOT\", \"exec_prefix\": \"$EXEC_PREFIX\", \"container\": \"$CN\", \"image\": \"$REF\", \"sqsh\": \"$SQSH\", \"cuda\": \"$CUDA_VER\", \"nvcc\": \"/usr/local/cuda/bin/nvcc\", \"nvcc_version\": \"$NVCC_VER\", \"cmake\": \"/opt/cmake/bin/cmake\", \"cmake_version\": \"$CMAKE_VER\", \"clone_commit\": \"$COMMIT\", \"driver\": \"$DRIVER\", \"gpu_name\": \"$GPU_NAME\", \"gpu_uuid\": \"$UUID\", \"gpu_index\": 0, \"concurrent_exec\": \"$CONC_OK\", \"smoke\": \"ok\"}"
