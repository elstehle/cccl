# Node workflow helpers for opt-W4 (one holder name, one process). Source this file, then:
#   w4_lease  [arch] [min-remaining] [wait]   # acquire the GPU lease, writes state/opt-W4/lease.txt
#   w4_target                                  # read the leased node into target.env
#   w4_ship                                    # harness (shared) plus W4 adapters and approaches/W4 from the W4 worktree
#   w4_build   <run.py args>                   # run.py --build-only (outside the GPU lock, under the lease)
#   w4_measure <run.py args>                   # run.py --measure-only (inside the GPU lock)
#   w4_run     <run.py args>                   # build then measure in one run.py invocation
#   w4_fetch                                   # results to state/results/opt-W4/
#   w4_release
# Every remote command keeps its output short and logs on the node.
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
W4WT=/Users/estehle/Downloads/cccl-topk-swarm-W4
W4=$W4WT/dsb_experiments/topk_swarm
HOLDER=opt-W4
STATE=$SWARM/state/opt-W4
mkdir -p $STATE $SWARM/state/results/$HOLDER

w4_lease() {
  python3 $SWARM/tools/lease.py acquire --holder $HOLDER --arch ${1:-b200} --min-remaining ${2:-40m} --wait ${3:-10m} | tee $STATE/lease.txt
}
w4_release() { python3 $SWARM/tools/lease.py release --holder $HOLDER; }

w4_target() {
  python3 - "$STATE/lease.txt" > $STATE/target.env <<'PY'
import sys, shlex
line = [l for l in open(sys.argv[1]).read().strip().splitlines() if l.startswith("node=")][-1]
kv = dict(tok.split("=", 1) for tok in shlex.split(line) if "=" in tok)
print("NODE=%s" % kv["node"])
print("ROOT=%s" % kv["root"])
print("EXEC_PREFIX=%s" % shlex.quote(kv.get("exec_prefix", "")))
print("ARCH=%s" % kv["arch"])
print("GPU=%s" % kv["gpu"])
print("NODE_NAME=%s" % kv["node"])
PY
  cat $STATE/target.env
}

w4_env() { source $STATE/target.env; }

w4_ssh() {
  w4_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}

w4_ship() {
  w4_env
  (cd $W4WT && git rev-parse HEAD > $W4/harness/SOURCE_COMMIT)
  w4_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $W4 harness approaches/W4 approaches/B3 | w4_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo shipped $(cat $W4/harness/SOURCE_COMMIT)
}

w4_runpy() {
  # $1 = log name, rest = run.py args
  w4_env
  local log=$1; shift
  w4_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu $GPU --node-name $NODE_NAME --arch-name $ARCH --cells ../approaches/W4/cells_w4.json $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; tail -n 4 $ROOT/work/$HOLDER/$log.log | cut -c1-220"
}

w4_build() { w4_runpy build_$(date +%H%M%S) --build-only "$@"; }
w4_measure() { w4_runpy measure_$(date +%H%M%S) --measure-only "$@"; }
w4_run() { w4_runpy run_$(date +%H%M%S) "$@"; }

w4_fetch() {
  w4_env
  w4_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -8
}
