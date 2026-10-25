#!/usr/bin/env bash
# Second-GB200 rerun of the flagged class (phase 3): decides whether the 10 percent excess of 128-thread
# kernels above 50 KB of SASS seen on gb200-nvl4-ts2-105 is a node component or a GB200 property.
#
#   harness/micro/gb200_rerun.sh <node> [<ts2-105 batch json or directory> ...]
#
# Outside the lease: ship the harness (B2_packed, B1_reference, N8_b2packed adapters, both cell grids) and
# icache_micro to the node (enroot host path) and build icache_micro. Under one pinned lease: the icache
# sweep (clocks, 80 KB rounds, L2 map), the B2_packed matrix (sanity cell plus the four discriminating
# cells) and the N8_b2packed flagged cells (128 threads, bf16 and fp32, both requirements, focus and
# flipbands grids, latency). Release, fetch, then print every flagged cell as a ratio to the ts2-105
# value found in the given batch files (eval-engineer's GB200 pass) or in my ts2-105 pre-flight matrix.
set -uo pipefail
N=${1:?node}
shift
REFS=("$@")
H=hw-researcher
SW=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
OUT=$SW/state/results/hw-researcher/gb200_rerun_$N
mkdir -p "$OUT"
cd "$SW" || exit 1
PREFIX=$(python3 - "$N" <<'EOF'
import json, sys
for n in json.load(open('state/nodes.json'))['nodes']:
    if n.get('ssh') == sys.argv[1] and n.get('status') in ('ready', 'expiring'):
        print(n.get('exec_prefix'))
EOF
)
[[ -n "$PREFIX" ]] || { echo "node $N not ready in the registry"; exit 1; }
HOSTROOT=/tmp/estehle/topk_swarm/topk_swarm
[[ "$PREFIX" == *enroot* ]] || { echo "this script expects an enroot (GB200) node, prefix is: $PREFIX"; exit 1; }
echo "$(date -u +%H:%M:%SZ) node $N"

# 1. ship and build outside the lease
(cd "$SW/harness" && tar czf - micro/icache_micro.cu micro/icache_bodies.h micro/run_icache.sh run.py topk_harness.cuh canary_main.cu cells_focus.json cells_flipbands.json adapters/B2_packed.cuh adapters/B2_detail.cuh adapters/B1_reference.cuh adapters/N8_b2packed.cuh) \
  | ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "mkdir -p $HOSTROOT/work/hw-researcher/rerun $HOSTROOT/results/hw-researcher && cd $HOSTROOT/work/hw-researcher/rerun && tar xzf - && chmod +x micro/run_icache.sh && mkdir -p micro/build_icache cell && mv -f run.py topk_harness.cuh canary_main.cu cells_focus.json cells_flipbands.json cell/ 2>/dev/null; rm -rf cell/adapters; mv adapters cell/; $PREFIX bash -c 'cd /work/topk_swarm/work/hw-researcher/rerun/micro && /usr/local/cuda/bin/nvcc -arch=sm_100 -O3 -std=c++17 -Xptxas -v -o build_icache/icache_micro icache_micro.cu > build_icache/ptxas_icache.txt 2>&1; echo build exit=\\$?'" 2>&1 | tail -1

# 2. lease, measure, release
L=$(python3 tools/lease.py acquire --holder "$H" --arch gb200 --node "$N" --min-remaining 20m --wait 30m 2>&1 | tail -1)
echo "$(date -u +%H:%M:%SZ) acquire: ${L:0:70}"
echo "$L" | grep -q "node=$N" || { echo "no lease on $N"; python3 tools/lease.py release --holder "$H" >/dev/null 2>&1; exit 2; }
T0=$(date +%s)
RUNPY="python3 run.py --repo /work/topk_swarm/cccl --harness /work/topk_swarm/work/hw-researcher/rerun/cell --holder $H --lock /work/topk_swarm/locks/gpu0.lock --gpu 0 --control B1_reference --regimes latency --patterns random --seeds 1 --no-battery --nvcc /usr/local/cuda/bin/nvcc"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "$PREFIX bash -c '
set -o pipefail
cd /work/topk_swarm/work/hw-researcher/rerun
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name,pci.bus_id,uuid,vbios_version,ecc.mode.current,clocks.max.sm,clocks.max.mem,power.limit,driver_version --format=csv > gpu_info.txt
cd micro && ./run_icache.sh --out build_icache --node $N --gpu 0 --lock /work/topk_swarm/locks/gpu0.lock --skip-build > run_wrapper.log 2>&1; echo icache_exit=\$?
cd ../cell && mkdir -p w_m o_m w_f o_f w_b o_b
timeout 600 $RUNPY --work \$PWD/w_m --out \$PWD/o_m --adapter B2_packed --keys bf16 --n 1024,2048,4096 --k 64 --threads 128,256 --req not_guaranteed > m.log 2>&1; echo matrix_exit=\$?
timeout 900 $RUNPY --work \$PWD/w_f --out \$PWD/o_f --adapter N8_b2packed --cells cells_focus.json --keys bf16,fp32 --n 1024,4096 --k 64,256 --threads 128 > f.log 2>&1; echo focus_exit=\$?
timeout 900 $RUNPY --work \$PWD/w_b --out \$PWD/o_b --adapter N8_b2packed --cells cells_flipbands.json --keys bf16,fp32 --n 8192 --k 16,32,64,128,256 --threads 128 > b.log 2>&1; echo flipbands_exit=\$?
grep -hE \"^  (bf16|fp32)\" m.log f.log b.log | cut -c1-110
mkdir -p /work/topk_swarm/results/hw-researcher/gb200_rerun_$N && cd /work/topk_swarm/work/hw-researcher/rerun && cp gpu_info.txt micro/build_icache/icache_*.jsonl micro/build_icache/clocks_during_*.txt cell/o_m/*.json cell/o_f/*.json cell/o_b/*.json cell/*.log /work/topk_swarm/results/hw-researcher/gb200_rerun_$N/ 2>/dev/null; true'" 2>&1 | tail -40
python3 tools/lease.py release --holder "$H" 2>&1 | tail -1
echo "lease held $(( $(date +%s) - T0 )) s"

# 3. fetch and compare with ts2-105
ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "tar czf - -C $HOSTROOT/results/hw-researcher gb200_rerun_$N" | tar xzf - -C "$SW/state/results/hw-researcher/" && echo "fetched $(ls "$OUT" | wc -l) files"
python3 -I - "$N" "$OUT" "$SW/state/results/hw-researcher/preflight_gb200-nvl4-ts2-105" ${REFS[@]+"${REFS[@]}"} <<'EOF'
import glob, json, os, sys
node, out, pre105 = sys.argv[1], sys.argv[2], sys.argv[3]
refs = sys.argv[4:]
def cells(paths):
    res = {}
    for p in paths:
        files = glob.glob(os.path.join(p, '**', '*.json'), recursive=True) if os.path.isdir(p) else [p]
        for f in files:
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if not isinstance(j, dict) or 'node' not in j:
                continue
            def walk(o):
                if isinstance(o, dict):
                    if o.get('regime') == 'latency' and 'raw' in o:
                        yield o
                    for v in o.values():
                        yield from walk(v)
                elif isinstance(o, list):
                    for v in o:
                        yield from walk(v)
            for c in walk(j):
                res_ = c['raw']['variant']['latency'].get('resources', {})
                key = (c['raw']['variant']['id'], c['key_type'], c['threads'], c['n'], c['k'], c['requirement'])
                res[key] = dict(v=c['value'], ctrl=c.get('control_value'), sass=res_.get('sass_instructions'), regs=res_.get('registers'), host=j['node'].get('hostname'))
    return res
new = cells([out])
old = cells([pre105] + refs)
rows = []
for key in sorted(new):
    a = new[key]; b = old.get(key)
    rows.append((key, a, b))
print(f"{'adapter':12s} {'keys':5s} {'thr':>4s} {'N':>6s} {'k':>4s} {'req':20s} {'sass':>6s} {node[-6:]:>8s} {'ts2-105':>8s} {'ratio':>6s}")
flag_ratios = []
for key, a, b in rows:
    r = (a['v'] / b['v']) if b else float('nan')
    big = key[2] == 128 and (a['sass'] or 0) > 3200
    if big and b:
        flag_ratios.append(r)
    print(f"{key[0]:12s} {key[1]:5s} {key[2]:4d} {key[3]:6d} {key[4]:4d} {key[5]:20s} {a['sass'] or 0:6d} {a['v']:8.0f} {b['v'] if b else float('nan'):8.0f} {r:6.3f}{'  <- flagged class' if big else ''}")
if flag_ratios:
    import statistics
    print(f"\nflagged class: {len(flag_ratios)} cells, ratio to ts2-105 min {min(flag_ratios):.3f} median {statistics.median(flag_ratios):.3f} max {max(flag_ratios):.3f}")
    print("conclusion: " + ("node component on ts2-105 (second GB200 is faster by the same margin the B200s are)" if statistics.median(flag_ratios) < 0.95 else "GB200 generation property (second GB200 agrees with ts2-105)" if abs(statistics.median(flag_ratios) - 1) < 0.03 else "mixed, see per-cell ratios"))
else:
    print("\nno ts2-105 reference values found for the flagged cells, pass the GB200 pass batch files as arguments")
EOF
