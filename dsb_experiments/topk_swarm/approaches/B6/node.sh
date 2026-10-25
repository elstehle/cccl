# Node workflow helpers for opt-B6 (one holder name, one process). Source this file, then:
#   b6_lease  [arch] [min-remaining] [wait]   # acquire the GPU lease, writes state/opt-B6/lease.txt
#   b6_target                                  # read the leased node into target.env
#   b6_ship                                    # harness, adapters and approaches/{B1,B3,B6} from the B6 worktree
#   b6_build   <run.py args>                   # run.py --build-only (outside the GPU lock, under the lease)
#   b6_measure <run.py args>                   # run.py --measure-only (inside the GPU lock)
#   b6_fetch                                   # results to state/results/opt-B6/
#   b6_release
# Every remote command keeps its output short and logs on the node.
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
B6=/Users/estehle/Downloads/cccl-topk-swarm-B6/dsb_experiments/topk_swarm
HOLDER=opt-B6
STATE=$SWARM/state/opt-B6
mkdir -p $STATE $SWARM/state/results/$HOLDER

b6_lease() {
  python3 $SWARM/tools/lease.py acquire --holder $HOLDER --arch ${1:-b200} --min-remaining ${2:-40m} --wait ${3:-10m} | tee $STATE/lease.txt
}
b6_release() { python3 $SWARM/tools/lease.py release --holder $HOLDER; }

b6_target() {
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

b6_env() { source $STATE/target.env; }

b6_ssh() {
  b6_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}

b6_ship() {
  b6_env
  (cd /Users/estehle/Downloads/cccl-topk-swarm-B6 && git rev-parse HEAD > $B6/harness/SOURCE_COMMIT)
  b6_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $B6 harness approaches/B1 approaches/B3 approaches/B6 | b6_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo shipped $(cat $B6/harness/SOURCE_COMMIT)
}

b6_runpy() {
  # $1 = log name, rest = run.py args
  b6_env
  local log=$1; shift
  b6_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu $GPU --node-name $NODE_NAME --arch-name $ARCH --cells cells_b6.json $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; tail -n 4 $ROOT/work/$HOLDER/$log.log | cut -c1-220"
}

b6_build() { b6_runpy build_$(date +%H%M%S) --build-only "$@"; }
b6_measure() { b6_runpy measure_$(date +%H%M%S) --measure-only "$@"; }
b6_run() { b6_runpy run_$(date +%H%M%S) "$@"; }

b6_fetch() {
  b6_env
  b6_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -8
}
