# Node workflow helpers for opt-W1. Source this file, then call the functions.
#   source approaches/W1/node.sh
#   w1_target umbriel-b200-091      # pick the node (reads the registry), no lease needed to compile
#   w1_ship                          # shared harness plus my adapters and approaches/W1 to the node
#   w1_sync_cccl                     # reset <root>/cccl to origin/exp/topk-primitives-swarm
#   w1_build  <run.py args>          # compiles outside the lock (run.py --build-only)
#   w1_lease  [arch] [min-remaining] # acquire the GPU lease, must land on the target node
#   w1_measure <run.py args>         # run.py --measure-only inside the GPU lock, logs on the node
#   w1_fetch                         # results to state/results/opt-W1/, then w1_release
# Every remote command keeps its output short and logs on the node.
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
W1=/Users/estehle/Downloads/cccl-topk-swarm-W1/dsb_experiments/topk_swarm
HOLDER=opt-W1
STATE=$SWARM/state/opt-W1
mkdir -p $STATE $SWARM/state/results/$HOLDER

w1_target() {
  python3 - "$1" > $STATE/target.env <<'PY'
import json, sys, shlex
reg = json.load(open("/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm/state/nodes.json"))
name = sys.argv[1]
for n in reg["nodes"]:
    if n["ssh"] == name or n["name"] == name:
        print("NODE=%s" % n["ssh"])
        print("ROOT=%s" % n["root"])
        print("EXEC_PREFIX=%s" % shlex.quote(n.get("exec_prefix") or ""))
        print("ARCH=%s" % n["arch"])
        print("NODE_NAME=%s" % n["name"])
        break
else:
    sys.exit("node %s not in registry" % name)
PY
  cat $STATE/target.env
}

w1_env() { source $STATE/target.env; }

w1_ssh() {
  w1_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}

w1_ship() {
  w1_env
  w1_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $SWARM harness | w1_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  tar czf - -C $W1 harness/adapters approaches/W1 | w1_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo shipped
}

w1_sync_cccl() {
  w1_ssh "cd $ROOT/cccl && git fetch -q origin exp/topk-primitives-swarm && git reset -q --hard origin/exp/topk-primitives-swarm && git rev-parse --short HEAD"
}

w1_runpy() {
  # $1 = log name, rest = run.py args
  w1_env
  local log=$1; shift
  w1_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu 0 --node-name $NODE_NAME --arch-name $ARCH $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; tail -n 3 $ROOT/work/$HOLDER/$log.log | cut -c1-200"
}

w1_build() { w1_runpy build_$(date +%H%M%S) --build-only "$@"; }
w1_measure() { w1_runpy measure_$(date +%H%M%S) --measure-only "$@"; }

w1_lease() {
  python3 $SWARM/tools/lease.py acquire --holder $HOLDER --arch ${1:-any} --min-remaining ${2:-30m} --wait ${3:-10m} | tee $STATE/lease.txt
}
w1_release() { python3 $SWARM/tools/lease.py release --holder $HOLDER; }

w1_fetch() {
  w1_env
  w1_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -5
}
