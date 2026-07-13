#!/usr/bin/env bash
# Run the decoupled-look-back benchmark set (select.if / scan.exclusive.sum / reduce.by_key)
# with the standard nvbench flags; tag json outputs with $1 (baseline|af).
set -euo pipefail
TAG="$1"
cd /build_lb
FLAGS="--device 0 --timeout 30 --stopping-criterion entropy --throttle-threshold 90 --throttle-recovery-delay 0.15"
./bin/cub.bench.select.if.base          $FLAGS -a 'Entropy=0.544' --json /build_lb/select_if.$TAG.json          | tail -2
./bin/cub.bench.scan.exclusive.sum.base $FLAGS                    --json /build_lb/scan_exsum.$TAG.json         | tail -2
./bin/cub.bench.reduce.by_key.base      $FLAGS -a 'MaxSegSize=8'  --json /build_lb/rbk.$TAG.json                | tail -2
echo "DONE $TAG"
