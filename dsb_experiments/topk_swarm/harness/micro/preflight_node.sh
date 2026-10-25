#!/usr/bin/env bash
# Pre-flight of the final evaluation node (phase 3).
#
#   harness/micro/preflight_node.sh <node> [<holder>]
#
# Outside the lease: ship icache_micro and the cell harness to <root>/work/hw-researcher on the node and
# build icache_micro. Under a lease pinned to the node (at most 10 minutes): nvidia-smi inventory and a
# 100 ms clock trace, the icache sweep (straight-line code, the 80 KB rounds, per-SM L2 map), and the
# sanity cell B2_packed bf16 128 threads N 4096 k 64 against B1_reference through the harness (which
# runs the drift canary first and last). Then release, fetch, and write
# state/results/hw-researcher/preflight_<node>.md with the verdict against the healthy reference values.
#
# Reference values (healthy B200s umbriel-b200-047, -068, -091, 2026-10-25, nvcc 13.4.92): 80 KB IMAD round
# 195.2 cycles, branchy IMAD round 212.6, dependent IMAD chain 4.0 to 4.1 cycles per instruction up to
# 128 KB, L2 hit 278 to 310 cycles per hop, sanity cell 6056 (this harness copy) with the control near
# 6050 to 6070 and the canary 32.0, SM clock 1950 to 1965 MHz under load, memory 3996 MHz. The same references
# apply to a GB200 node for every in-core check (T16: identical per-SM cycle costs), only the SM clock probe
# may read lower (1951 to 1953 MHz seen on gb200-nvl4-ts2-91) and the L2 hit latency may differ by a few percent.
# A GB200 node needs its exec_prefix from the registry (enroot), which this script reads, and the host paths
# /tmp/estehle/topk_swarm/topk_swarm for the fetch are mapped to /work/topk_swarm inside the container.
set -uo pipefail
N=${1:?node}
H=${2:-hw-researcher}
SW=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
MICRO=$SW/harness/micro
OUT=$SW/state/results/hw-researcher/preflight_$N
mkdir -p "$OUT"
cd "$SW" || exit 1
PREFIX=$(python3 - "$N" <<'EOF'
import json, sys
for n in json.load(open('state/nodes.json'))['nodes']:
    if n.get('ssh') == sys.argv[1] and n.get('status') in ('ready', 'expiring'):
        print(n.get('exec_prefix') or 'docker exec -i topk_swarm')
EOF
)
[[ -n "$PREFIX" ]] || { echo "node $N not in the registry as ready"; exit 1; }
echo "$(date -u +%H:%M:%SZ) node $N prefix: ${PREFIX:0:40}"
# docker nodes have the root only inside the container, enroot nodes (GB200) map /work to a host path
if [[ "$PREFIX" == *enroot* ]]; then HOSTROOT=/tmp/estehle/topk_swarm/topk_swarm; else HOSTROOT=""; fi
ship_and_build() {
  if [[ -n "$HOSTROOT" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "mkdir -p $HOSTROOT/work/hw-researcher/preflight $HOSTROOT/results/hw-researcher && cd $HOSTROOT/work/hw-researcher/preflight && tar xzf - && chmod +x micro/run_icache.sh && mkdir -p micro/build_icache cell && mv -f run.py topk_harness.cuh canary_main.cu cells_focus.json cell/ 2>/dev/null; rm -rf cell/adapters; mv adapters cell/; $PREFIX bash -c 'cd /work/topk_swarm/work/hw-researcher/preflight/micro && /usr/local/cuda/bin/nvcc -arch=sm_100 -O3 -std=c++17 -Xptxas -v -o build_icache/icache_micro icache_micro.cu > build_icache/ptxas_icache.txt 2>&1; echo build exit=\$?'"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "$PREFIX bash -c 'mkdir -p /work/topk_swarm/work/hw-researcher/preflight && cd /work/topk_swarm/work/hw-researcher/preflight && tar xzf - && chmod +x micro/run_icache.sh && mkdir -p micro/build_icache cell && mv -f run.py topk_harness.cuh canary_main.cu cells_focus.json cell/ 2>/dev/null; rm -rf cell/adapters; mv adapters cell/; cd micro && /usr/local/cuda/bin/nvcc -arch=sm_100 -O3 -std=c++17 -Xptxas -v -o build_icache/icache_micro icache_micro.cu > build_icache/ptxas_icache.txt 2>&1; echo build exit=\$?'"
  fi
}
fetch_results() {
  if [[ -n "$HOSTROOT" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "tar czf - -C $HOSTROOT/results/hw-researcher preflight_$N"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "$PREFIX tar czf - -C /work/topk_swarm/results/hw-researcher preflight_$N"
  fi
}

# 1. ship and build outside the lease
(cd "$SW/harness" && tar czf - micro/icache_micro.cu micro/icache_bodies.h micro/run_icache.sh run.py topk_harness.cuh canary_main.cu cells_focus.json adapters/B2_packed.cuh adapters/B1_reference.cuh adapters/B2_detail.cuh) | ship_and_build 2>&1 | tail -2

# 2. lease (pinned, short), measure, release
L=$(python3 tools/lease.py acquire --holder "$H" --arch "${ARCH:-b200}" --node "$N" --min-remaining 15m --wait 20m 2>&1 | tail -1)
echo "$(date -u +%H:%M:%SZ) acquire: ${L:0:70}"
echo "$L" | grep -q "node=$N" || { echo "no lease on $N"; python3 tools/lease.py release --holder "$H" >/dev/null 2>&1; exit 2; }
GPU=$(echo "$L" | sed -n 's/.*gpu=\([0-9]*\).*/\1/p')
T0=$(date +%s)
ssh -o BatchMode=yes -o ConnectTimeout=20 "$N" "$PREFIX bash -c '
set -o pipefail
cd /work/topk_swarm/work/hw-researcher/preflight
export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi --query-gpu=name,pci.bus_id,uuid,vbios_version,ecc.mode.current,clocks.max.sm,clocks.max.mem,power.limit,driver_version,clocks_throttle_reasons.active --format=csv > gpu_info.txt
cd micro && ./run_icache.sh --out build_icache --node $N --gpu $GPU --lock /work/topk_swarm/locks/gpu$GPU.lock --skip-build > run_wrapper.log 2>&1; echo icache_exit=\$?
cd ../cell && mkdir -p work out && (nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader -lms 100 > clocks_cell.txt 2>/dev/null & echo \$! > smi.pid)
timeout 480 python3 run.py --repo /work/topk_swarm/cccl --harness /work/topk_swarm/work/hw-researcher/preflight/cell --holder $H --work /work/topk_swarm/work/hw-researcher/preflight/cell/work --out /work/topk_swarm/work/hw-researcher/preflight/cell/out --lock /work/topk_swarm/locks/gpu$GPU.lock --gpu $GPU --adapter B2_packed --control B1_reference --keys bf16 --n 4096 --k 64 --threads 128 --req not_guaranteed --regimes latency --patterns random --seeds 1 --no-battery --nvcc /usr/local/cuda/bin/nvcc > cell_run.log 2>&1; echo cell_exit=\$?
kill \$(cat smi.pid) 2>/dev/null; rm -f smi.pid
grep -E \"^  bf16|cells with\" cell_run.log | cut -c1-110
cd /work/topk_swarm/work/hw-researcher/preflight && mkdir -p /work/topk_swarm/results/hw-researcher/preflight_$N && cp gpu_info.txt micro/build_icache/icache_*.jsonl micro/build_icache/clocks_during_*.txt micro/build_icache/gpu_info_*.txt cell/clocks_cell.txt cell/out/*.json cell/cell_run.log /work/topk_swarm/results/hw-researcher/preflight_$N/ 2>/dev/null; true'" 2>&1 | tail -8
python3 tools/lease.py release --holder "$H" 2>&1 | tail -1
echo "lease held $(( $(date +%s) - T0 )) s"

# 3. fetch and evaluate
fetch_results | tar xzf - -C "$SW/state/results/hw-researcher/" && echo "fetched $(ls "$OUT" | wc -l) files"
python3 -I - "$N" "$OUT" <<'EOF'
import glob, json, os, sys, statistics
node, out = sys.argv[1], sys.argv[2]
checks = []
def check(name, ok, detail):
    checks.append((name, ok, detail))
# inventory and clocks
info = open(os.path.join(out, 'gpu_info.txt')).read().strip().split('\n')[-1] if os.path.exists(os.path.join(out, 'gpu_info.txt')) else 'missing'
clk = []
for f in glob.glob(os.path.join(out, 'clocks_*.txt')):
    for l in open(f):
        parts = [p.strip() for p in l.split(',')]
        if len(parts) >= 3 and parts[1].endswith('MHz'):
            try:
                clk.append((int(parts[1].split()[0]), int(parts[2].split()[0])))
            except ValueError:
                pass
sm_busy = sorted(c[0] for c in clk if c[0] > 1500)
mem = sorted(set(c[1] for c in clk))
check('SM clock under load', bool(sm_busy) and sm_busy[0] >= 1900, f"{sm_busy[0] if sm_busy else 'none'} to {sm_busy[-1] if sm_busy else 'none'} MHz over {len(sm_busy)} busy samples, memory {mem}")
# icache rows
fs = sorted(glob.glob(os.path.join(out, 'icache_*.jsonl')), key=lambda f: f.split('_')[-1])
rows = [json.loads(l) for l in open(fs[-1]) if l.strip()] if fs else []
def row(prefix):
    for j in rows:
        if j['name'].startswith(prefix):
            return j
probe = [round(j['mhz']) for j in rows if 'mhz' in j]
check('SM clock probe in kernel', bool(probe) and min(probe) >= 1900, f"{probe}")
r = row('rounds one block ROUNDS 128 BLOCK 128 ')
check('80 KB IMAD round', r is not None and abs(r['per_round_min'] - 195.2) <= 2.0, f"{r['per_round_min'] if r else 'missing'} cycles per round (reference 195.2)")
rb = row('rounds branchy-imad one block ROUNDS 128 BLOCK 128 ')
check('83 KB branchy IMAD round', rb is not None and abs(rb['per_round_min'] - 212.6) <= 2.5, f"{rb['per_round_min'] if rb else 'missing'} (reference 212.6)")
rall = row('rounds one block per SM ROUNDS 128 BLOCK 128 ')
check('80 KB round on all SMs', rall is not None and rall['per_round_max'] - rall['per_round_min'] <= 2.0, f"{rall['per_round_min'] if rall else '?'} to {rall['per_round_max'] if rall else '?'} over {len(rall['per_block']) if rall else 0} SMs")
dep = {(j['len'] * 16 // 1024, j['block']): j['cycles_per_instr'] for j in rows if j.get('mode') == 'dependent'}
check('dependent IMAD chain 128 KB at 128 threads', dep.get((128, 128), 9) <= 4.2, f"{dep.get((16,128))}, {dep.get((64,128))}, {dep.get((128,128))} cycles per instruction at 16, 64, 128 KB")
c1 = row('chase L2 4MB blocks 1')
check('L2 hit latency on the landing SM', c1 is not None and 270 <= c1['per_hop_min'] <= 315, f"{c1['per_hop_min'] if c1 else 'missing'} cycles per hop on SM {c1['per_block'][0][0] if c1 else '?'} (healthy 278 to 310)")
# sanity cell
cell = None
for f in glob.glob(os.path.join(out, '*.json')):
    j = json.load(open(f))
    if 'node' not in j:
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
        if c['threads'] == 128 and c['n'] == 4096 and c['k'] == 64:
            cell = c
if cell:
    lat = cell['raw']['variant']['latency']
    check('sanity cell B2_packed 128 thr N 4096 k 64', 5900 <= cell['value'] <= 6300, f"{cell['value']:.0f} cycles (healthy 6056 to 6254), control {cell.get('control_value') or 0:.0f} (healthy 6050 to 6070), chain minima {lat['chain_minima']}")
    check('harness canary', abs((cell.get('canary_start') or 0) - 32.0) < 0.5 and abs((cell.get('canary_end') or 0) - 32.0) < 0.5, f"start {cell.get('canary_start')} end {cell.get('canary_end')} (reference 32.0)")
else:
    check('sanity cell B2_packed 128 thr N 4096 k 64', False, 'no cell record fetched')
ok = all(c[1] for c in checks)
lines = [f"# Pre-flight of {node} for the phase 3 evaluation", "", f"Verdict: {'passed' if ok else 'FAILED'}. Measured by hw-researcher on {os.popen('date -u +%Y-%m-%dT%H:%MZ').read().strip()} under a pinned lease, files in state/results/hw-researcher/preflight_{node}/.", "", f"GPU: {info}", ""]
for name, good, detail in checks:
    lines.append(f"- {'ok ' if good else 'BAD'} {name}: {detail}")
lines += ["", "Reference values from the healthy B200s umbriel-b200-047, -068 and -091 (2026-10-25, nvcc 13.4.92): 80 KB IMAD round 195.2 cycles, branchy 212.6, dependent IMAD chain 4.0 to 4.1 per instruction up to 128 KB, L2 hit 278 to 310 cycles per hop, sanity cell 6056 to 6254 with the B1_reference control near 6050 to 6070 and the canary 32.0, SM clock 1950 to 1965 MHz under load, memory 3996 MHz. The retired outlier umbriel-b200-020 gave 7831 on the sanity cell (T58)."]
open(os.path.join(os.path.dirname(out), f'preflight_{node}.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
EOF
