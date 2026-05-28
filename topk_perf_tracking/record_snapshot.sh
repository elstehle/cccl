#!/usr/bin/env bash
# Record a new snapshot of TopK kernel resource usage on a B200 node.
#
# Usage:
#   record_snapshot.sh LABEL [BRANCH_OR_REF]
#
# LABEL is the snapshot label, used to name the JSON snapshot file.
# BRANCH_OR_REF is the local commit-ish to evaluate. Defaults to HEAD. The
# script pushes the ref to `origin/tmp/perf-eval-<label>` so the container can
# fetch and check it out, builds the pairs benchmark with PTXAS verbose
# output, copies the build log back to the host, parses it, and writes the
# snapshot to `snapshots/<label>.json`.
#
# Node configuration (override via env vars to switch B200 nodes):
#   TOPK_HOST            -- ssh-reachable B200 hostname
#                             (default: umbriel-b200-068)
#   TOPK_CONTAINER       -- name of the running Docker container with the
#                             CCCL clone (default: clever_hellman)
#   TOPK_REPO_DIR        -- path to the CCCL clone inside the container
#                             (default: /cccl/cccl -- NFS-mounted on the
#                             current node; on previous nodes this was
#                             /cccl_fork/cccl on the overlay disk)
#   TOPK_BUILD_DIR_BASE  -- scratch dir inside the container for build
#                             trees. MUST be on the local overlay disk on
#                             umbriel-b200-068 since /cccl is NFS with
#                             ~250 MB free (default: /cccl_fork/topk_perf)
#   TOPK_CMAKE_BIN       -- cmake >= 4.0 binary inside the container
#                             (default: /cccl/cmake/cmake-4.3.2-linux-x86_64/bin/cmake)
#   TOPK_NINJA_JOBS      -- parallelism passed to ninja (default: 16)
#
# Known-good node configurations (latest first):
#   umbriel-b200-068 / clever_hellman  / repo=/cccl/cccl       (NFS)
#   umbriel-b200-072 / bold_mahavira   / repo=/cccl_fork/cccl  (overlay)
#   umb-b200-261     / brave_rosalind  / repo=/cccl_fork/cccl  (overlay)
#
# The container is expected to have `nvidia/cuda:...-devel` style tools
# (`nvcc`, `cuobjdump`, `nvdisasm`) on PATH and `origin` of the CCCL
# clone pointing at the elstehle/cccl fork.

set -euo pipefail

LABEL="${1:?usage: $0 LABEL [BRANCH_OR_REF]}"
REF="${2:-HEAD}"

HOST="${TOPK_HOST:-umbriel-b200-068}"
CONTAINER="${TOPK_CONTAINER:-clever_hellman}"
REPO_DIR_IN_CONTAINER="${TOPK_REPO_DIR:-/cccl/cccl}"
BUILD_DIR_BASE="${TOPK_BUILD_DIR_BASE:-/cccl_fork/topk_perf}"
CMAKE_BIN="${TOPK_CMAKE_BIN:-/cccl/cmake/cmake-4.3.2-linux-x86_64/bin/cmake}"
NINJA_JOBS="${TOPK_NINJA_JOBS:-16}"

BUILD_DIR="$BUILD_DIR_BASE/build_${LABEL}"
LOG_PATH="$BUILD_DIR_BASE/build_${LABEL}_pairs.log"
TMP_REF="refs/heads/tmp/perf-eval-${LABEL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Resolving ref: $REF"
SHA=$(cd "$REPO_DIR" && git rev-parse "$REF")
SUBJECT=$(cd "$REPO_DIR" && git log -1 --format='%s' "$REF")
echo "    sha: $SHA"
echo "    subject: $SUBJECT"

echo "==> Pushing $REF to origin:$TMP_REF"
(cd "$REPO_DIR" && git push --force origin "$SHA:$TMP_REF")

echo "==> Fetching and checking out on container"
ssh "$HOST" "docker exec $CONTAINER bash -lc \"\
  cd $REPO_DIR_IN_CONTAINER && \
  git fetch origin tmp/perf-eval-${LABEL} && \
  git checkout -B tmp/perf-eval-${LABEL} origin/tmp/perf-eval-${LABEL} && \
  git log -1 --oneline\""

echo "==> Configuring"
ssh "$HOST" "docker exec $CONTAINER bash -lc \"\
  export PATH=$(dirname $CMAKE_BIN):\\\$PATH && \
  cd $REPO_DIR_IN_CONTAINER && \
  rm -rf $BUILD_DIR && \
  mkdir -p $BUILD_DIR_BASE && \
  $CMAKE_BIN -G Ninja -B $BUILD_DIR --preset cub-cpp17 \
    -DCCCL_ENABLE_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DCMAKE_CUDA_FLAGS='-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t' \
  2>&1 | tail -5\""

echo "==> Building cub.bench.topk.pairs.base"
START_TS=$(date +%s)
ssh "$HOST" "docker exec $CONTAINER bash -lc \"\
  cd $BUILD_DIR && \
  ninja cub.bench.topk.pairs.base -j $NINJA_JOBS > $LOG_PATH 2>&1 && \
  echo DONE\""
END_TS=$(date +%s)
DELTA=$(( END_TS - START_TS ))
echo "    build wall-time: ${DELTA}s"

echo "==> Pulling build log"
mkdir -p "$SCRIPT_DIR/raw_logs"
LOCAL_LOG="$SCRIPT_DIR/raw_logs/${LABEL}__pairs.log"
ssh "$HOST" "docker exec $CONTAINER cat $LOG_PATH" > "$LOCAL_LOG"
echo "    wrote $LOCAL_LOG"

echo "==> Parsing into snapshot"
CTK_VERSION=$(ssh "$HOST" "docker exec $CONTAINER bash -lc 'nvcc --version | grep release | awk -F\" V\" \"{print \\\$2}\"'" | tr -d '[:space:]')
python3 "$SCRIPT_DIR/build_snapshot.py" "$LOCAL_LOG" \
  --label "$LABEL" \
  --branch "$REF" \
  --sha "$SHA" \
  --commit-subject "$SUBJECT" \
  --build-target cub.bench.topk.pairs.base \
  --build-seconds "$DELTA" \
  --gpu-arch sm_100 \
  --ctk-version "$CTK_VERSION" \
  --build-flags '-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t'

echo "==> Done"
echo "    snapshot: $SCRIPT_DIR/snapshots/${LABEL}.json"
