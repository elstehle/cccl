#!/bin/bash
# DeviceMergeSort benchmark comparison: /cccl_main (NVIDIA main) vs /cccl_fix (PR #10733).
# Runs INSIDE the container: cat bms_bench_run.sh | docker exec -i <ctr> bash
echo ===TOOLING===
command -v cmake >/dev/null || pip install -q --break-system-packages cmake 2>&1 | tail -1
command -v ninja >/dev/null || pip install -q --break-system-packages ninja 2>&1 | tail -1
cmake --version | head -1
ninja --version 2>/dev/null

echo ===CONFIGURE===
# benchmarks require Thrust (nvbench_helper data generation) -> do NOT disable components;
# wipe stale build dirs so old cache values (THRUST=OFF) cannot linger
rm -rf /b_main /b_fix
# CCCL_ENABLE_CUB=ON is required: without it cub/ configures in header-only package mode and
# returns before the benchmark gate (CCCL_ENABLE_CUB defaults OFF)
cmake -S /cccl_main -B /b_main -G Ninja -DCCCL_ENABLE_CUB=ON -DCCCL_ENABLE_BENCHMARKS=ON \
  -DCUB_ENABLE_TESTING=OFF -DCUB_ENABLE_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=100 > /root/cfg_main.log 2>&1 && echo cfg_main OK || tail -30 /root/cfg_main.log
cmake -S /cccl_fix -B /b_fix -G Ninja -DCCCL_ENABLE_CUB=ON -DCCCL_ENABLE_BENCHMARKS=ON \
  -DCUB_ENABLE_TESTING=OFF -DCUB_ENABLE_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=100 > /root/cfg_fix.log 2>&1 && echo cfg_fix OK || tail -30 /root/cfg_fix.log
echo "bench targets found: $(ninja -C /b_main -t targets all 2>/dev/null | grep -c 'cub\.bench')"
ninja -C /b_main -t targets all 2>/dev/null | grep -i 'merge_sort' | head -5

echo ===BUILD===
cmake --build /b_main --target cub.bench.merge_sort.keys.base cub.bench.merge_sort.pairs.base \
  > /root/bld_main.log 2>&1 && echo bld_main OK \
  || { tail -25 /root/bld_main.log; ninja -C /b_main -t targets all 2>/dev/null | grep -i merge_sort | head; }
cmake --build /b_fix --target cub.bench.merge_sort.keys.base cub.bench.merge_sort.pairs.base \
  > /root/bld_fix.log 2>&1 && echo bld_fix OK || tail -25 /root/bld_fix.log

echo ===RUN===
/b_main/bin/cub.bench.merge_sort.keys.base -d 0 --json /root/keys_main.json > /root/run_km.log 2>&1 \
  && echo keys_main OK || tail -15 /root/run_km.log
/b_fix/bin/cub.bench.merge_sort.keys.base -d 0 --json /root/keys_fix.json > /root/run_kf.log 2>&1 \
  && echo keys_fix OK || tail -15 /root/run_kf.log
/b_main/bin/cub.bench.merge_sort.pairs.base -d 0 --json /root/pairs_main.json > /root/run_pm.log 2>&1 \
  && echo pairs_main OK || tail -15 /root/run_pm.log
/b_fix/bin/cub.bench.merge_sort.pairs.base -d 0 --json /root/pairs_fix.json > /root/run_pf.log 2>&1 \
  && echo pairs_fix OK || tail -15 /root/run_pf.log

echo ===COMPARE===
pip install -q --break-system-packages tabulate 2>/dev/null
CMP=$(find /b_main -name nvbench_compare.py 2>/dev/null | head -1)
echo "script=$CMP"
echo "--- KEYS: main vs PR ---"
python3 "$CMP" /root/keys_main.json /root/keys_fix.json 2>&1 | tail -60
echo "--- PAIRS: main vs PR ---"
python3 "$CMP" /root/pairs_main.json /root/pairs_fix.json 2>&1 | tail -60
