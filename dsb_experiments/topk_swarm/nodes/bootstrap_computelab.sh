#!/usr/bin/env bash
# Bootstrap a computelab B200 node (x86_64, docker) for topk_swarm. Runs ON the node as the user.
#   bootstrap_computelab.sh <gpu-uuid> [image]
# Creates a detached container `topk_swarm` from a CUDA devel image with only the job's GPU
# (--gpus device=<uuid>, so it is CUDA index 0 inside), binds /tmp/topk_swarm at /work, installs
# git, python3, ninja, copies the extracted cmake from the NFS home, clones the fork branch
# blobless, and runs a trivial sm_100 kernel under the gpu0 lock. Idempotent. Prints short
# progress lines and a final `BOOTSTRAP_JSON {...}` line for tools/broker.py. Details in
# /tmp/topk_swarm/topk_swarm/logs/bootstrap.log on the node.
set -u
UUID=${1:?gpu uuid}
IMAGE=${2:-nvidia/cuda:13.4.2-devel-ubuntu24.04}
BRANCH=exp/topk-primitives-swarm
REPO=https://github.com/elstehle/cccl.git
HOST_ROOT=/tmp/topk_swarm        # bind-mounted at /work inside the container
ROOT=/work/topk_swarm            # root as seen through exec_prefix
CN=topk_swarm
NFS_CMAKE=$HOME/cccl/cmake/cmake-4.3.2-linux-x86_64

mkdir -p "$HOST_ROOT/topk_swarm"/{locks,work,results,logs} || { echo "BOOTSTRAP_FAIL mkdir $HOST_ROOT"; exit 1; }
LOG=$HOST_ROOT/topk_swarm/logs/bootstrap.log
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { step "BOOTSTRAP_FAIL $*"; echo "--- log tail"; tail -n 12 "$LOG"; exit 1; }
dx() { docker exec -i "$CN" bash -c "$*"; }

step "bootstrap start host=$(hostname) uuid=$UUID image=$IMAGE"
command -v flock >/dev/null || fail "no flock on host"
command -v docker >/dev/null || fail "no docker on host"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  step "pulling $IMAGE"
  docker pull "$IMAGE" >>"$LOG" 2>&1 || fail "docker pull $IMAGE"
fi

state=$(docker inspect -f '{{.State.Status}}' "$CN" 2>/dev/null | tr -d '[:space:]')
[ -n "$state" ] || state=none
if [ "$state" = none ]; then
  step "starting container $CN"
  docker run -d --name "$CN" --init --gpus "\"device=$UUID\"" -v "$HOST_ROOT":/work -w /work \
    "$IMAGE" sleep infinity >>"$LOG" 2>&1 || fail "docker run"
elif [ "$state" != running ]; then
  step "restarting container $CN (was $state)"
  docker start "$CN" >>"$LOG" 2>&1 || fail "docker start"
else
  step "reusing running container $CN"
fi
GPU_LINE=$(dx 'nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader' 2>>"$LOG") || fail "nvidia-smi in container"
NGPU=$(printf '%s\n' "$GPU_LINE" | grep -c .)
[ "$NGPU" = 1 ] || fail "expected exactly 1 GPU in container, got $NGPU: $GPU_LINE"
printf '%s\n' "$GPU_LINE" | grep -q "$UUID" || fail "container GPU is not $UUID: $GPU_LINE"
step "container gpu: $GPU_LINE"

step "packages"
dx 'export DEBIAN_FRONTEND=noninteractive; (command -v git && command -v python3 && command -v ninja && command -v flock) >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 python3-venv python3-pip ninja-build util-linux ca-certificates less procps)' >>"$LOG" 2>&1 || fail "apt-get"
dx 'command -v flock' >/dev/null 2>&1 || fail "no flock in container"

step "cmake"
if [ ! -x "$HOST_ROOT/cmake/bin/cmake" ]; then
  if [ -x "$NFS_CMAKE/bin/cmake" ]; then
    cp -r "$NFS_CMAKE" "$HOST_ROOT/cmake" >>"$LOG" 2>&1 || fail "cmake copy from NFS"
  else
    dx 'python3 -m venv /work/venv && /work/venv/bin/pip install -q cmake ninja && mkdir -p /work/cmake && ln -sfn /work/venv/bin /work/cmake/bin' >>"$LOG" 2>&1 || fail "pip cmake"
  fi
fi
dx 'ln -sfn /work/cmake /opt/cmake && ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake' >>"$LOG" 2>&1 || fail "cmake symlink"
CMAKE_VER=$(dx 'cmake --version | head -1 | grep -o "[0-9][0-9.a-z-]*"' 2>>"$LOG") || fail "cmake --version"

step "clone $BRANCH"
dx "test -d $ROOT/cccl/.git || git clone -q --filter=blob:none -b $BRANCH $REPO $ROOT/cccl" >>"$LOG" 2>&1 || fail "git clone"
dx "cd $ROOT/cccl && git fetch -q origin $BRANCH && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH" >>"$LOG" 2>&1 || fail "git update"
COMMIT=$(dx "git -C $ROOT/cccl rev-parse --short=10 HEAD" 2>>"$LOG") || fail "rev-parse"

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
dx "cd $ROOT/logs && nvcc -arch=sm_100 -O3 -o smoke smoke.cu" >>"$LOG" 2>&1 || fail "nvcc smoke.cu"
SMOKE=$(dx "CUDA_VISIBLE_DEVICES=0 flock $ROOT/locks/gpu0.lock -c $ROOT/logs/smoke" 2>>"$LOG") || fail "smoke run: $SMOKE"
echo "$SMOKE" >>"$LOG"
printf '%s\n' "$SMOKE" | grep -q SMOKE_OK || fail "smoke output: $SMOKE"
step "$SMOKE"

NVCC_VER=$(dx 'nvcc --version | grep -o "V[0-9][0-9.]*" | head -1')
CUDA_VER=$(dx "grep -o '\"version\" *: *\"[0-9.]*\"' /usr/local/cuda/version.json 2>/dev/null | head -1 | grep -o '[0-9][0-9.]*'")
[ -n "$CUDA_VER" ] || CUDA_VER=${NVCC_VER#V}
DRIVER=$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $4}')
GPU_NAME=$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $3}')
step "done cuda=$CUDA_VER nvcc=$NVCC_VER cmake=$CMAKE_VER driver=$DRIVER commit=$COMMIT"
echo "BOOTSTRAP_JSON {\"root\": \"$ROOT\", \"host_root\": \"$HOST_ROOT\", \"exec_prefix\": \"docker exec -i $CN\", \"container\": \"$CN\", \"image\": \"$IMAGE\", \"cuda\": \"$CUDA_VER\", \"nvcc\": \"/usr/local/cuda/bin/nvcc\", \"nvcc_version\": \"$NVCC_VER\", \"cmake\": \"/opt/cmake/bin/cmake\", \"cmake_version\": \"$CMAKE_VER\", \"clone_commit\": \"$COMMIT\", \"driver\": \"$DRIVER\", \"gpu_name\": \"$GPU_NAME\", \"gpu_uuid\": \"$UUID\", \"gpu_index\": 0, \"smoke\": \"ok\"}"
