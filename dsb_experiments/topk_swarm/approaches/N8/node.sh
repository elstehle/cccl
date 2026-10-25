# Node workflow helpers for opt-N8 (one holder name, one process). Source this file, then:
#   n8_lease  [arch] [min-remaining] [wait]   # acquire the GPU lease, writes state/opt-N8/lease.txt
#   n8_target                                  # read the leased node into target.env
#   n8_ship                                    # harness, adapters and approaches/{B1,B3,N8} from the N8 worktree
#   n8_build   <run.py args>                   # run.py --build-only (outside the GPU lock, under the lease)
#   n8_measure <run.py args>                   # run.py --measure-only (inside the GPU lock)
#   n8_fetch                                   # results to state/results/opt-N8/
#   n8_release
# Every remote command keeps its output short and logs on the node.
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
N8=/Users/estehle/Downloads/cccl-topk-swarm-N8/dsb_experiments/topk_swarm
HOLDER=opt-N8
STATE=$SWARM/state/opt-N8
mkdir -p $STATE $SWARM/state/results/$HOLDER

n8_lease() {
  python3 $SWARM/tools/lease.py acquire --holder $HOLDER --arch ${1:-b200} --min-remaining ${2:-40m} --wait ${3:-10m} | tee $STATE/lease.txt
}
n8_release() { python3 $SWARM/tools/lease.py release --holder $HOLDER; }

n8_target() {
  python3 - "$STATE/lease.txt" > $STATE/target.env <<'PY'
import sys, shlex
line = open(sys.argv[1]).read().strip().splitlines()[-1]
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

n8_env() { source $STATE/target.env; }

n8_ssh() {
  n8_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}

n8_ship() {
  n8_env
  n8_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $N8 harness approaches/B1 approaches/B3 approaches/N8 | n8_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo shipped
}

n8_runpy() {
  # $1 = log name, rest = run.py args
  n8_env
  local log=$1; shift
  n8_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu $GPU --node-name $NODE_NAME --arch-name $ARCH --cells cells_n8.json $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; tail -n 3 $ROOT/work/$HOLDER/$log.log | cut -c1-200"
}

n8_build() { n8_runpy build_$(date +%H%M%S) --build-only "$@"; }
n8_measure() { n8_runpy measure_$(date +%H%M%S) --measure-only "$@"; }

n8_fetch() {
  n8_env
  n8_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -8
}
