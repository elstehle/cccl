# Node workflow for N5 (holder opt-N5). Source, then call.
#   n5_target <node>      pick a registry node (ssh name) and cache its root, exec prefix and arch
#   n5_lease              acquire a lease through tools/lease.py and target the node it hands out
#   n5_release            release the lease
#   n5_ship               tar the worktree's harness (adapters included) and approaches to the node
#   n5_build <tag> <args> run.py --build-only (outside the lock), log tail
#   n5_measure <tag> <a>  run.py --measure-only (inside the lock), log tail
#   n5_run <tag> <args>   build and measure in one run.py
#   n5_fetch              fetch the node's results into state/results/opt-N5
#   n5_sass <gen> <cell>  opcode histogram of one cell binary (kernel name filter optional)
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
WT=/Users/estehle/Downloads/cccl-topk-swarm-N5
WS=$WT/dsb_experiments/topk_swarm
HOLDER=opt-N5
STATE=$SWARM/state/$HOLDER
mkdir -p $STATE $SWARM/state/results/$HOLDER

n5_target() {
  python3 - "$1" > $STATE/target.env <<'PY'
import json, sys, shlex
reg = json.load(open("/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm/state/nodes.json"))
name = sys.argv[1]
for n in reg["nodes"]:
    if (n["ssh"] == name or n["name"] == name) and n.get("status") in ("ready", "expiring"):
        print("NODE=%s" % n["ssh"])
        print("ROOT=%s" % n["root"])
        print("EXEC_PREFIX=%s" % shlex.quote(n.get("exec_prefix") or ""))
        print("ARCH=%s" % n["arch"])
        print("NODE_NAME=%s" % n["name"])
        break
else:
    sys.exit("node %s not ready in registry" % name)
PY
  cat $STATE/target.env
}
n5_lease() {
  local line
  line=$(python3 $SWARM/tools/lease.py acquire --holder $HOLDER --arch "${1:-b200}" --min-remaining "${2:-60m}" --wait "${3:-45m}") || { echo "$line"; return 1; }
  echo "$line" | tee $STATE/lease.txt
  local node
  node=$(echo "$line" | sed -n 's/^node=\([^ ]*\).*/\1/p')
  n5_target "$node"
}
n5_release() { python3 $SWARM/tools/lease.py release --holder $HOLDER; }
n5_env() { source $STATE/target.env; }
n5_ssh() {
  n5_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}
n5_ship() {
  n5_env
  git -C $WT rev-parse HEAD > $WS/harness/SOURCE_COMMIT
  n5_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $WS harness approaches/W1 approaches/N5 | n5_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo "shipped $(cat $WS/harness/SOURCE_COMMIT | cut -c1-10)"
}
n5_runpy() {
  n5_env
  local log=$1; shift
  n5_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu 0 --node-name $NODE_NAME --arch-name $ARCH --source-commit $(git -C $WT rev-parse --short=10 HEAD) $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; tail -n 4 $ROOT/work/$HOLDER/$log.log | cut -c1-240"
}
n5_build() { n5_runpy build_$1 --build-only "${@:2}"; }
n5_measure() { n5_runpy measure_$1 --measure-only "${@:2}"; }
n5_run() { n5_runpy run_$1 "${@:2}"; }
n5_fetch() {
  n5_env
  n5_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -6
}
n5_sass() {
  n5_env
  n5_ssh "cd $ROOT/work/$HOLDER/gen/$1 && /usr/local/cuda/bin/cuobjdump -sass $2 | awk -f $ROOT/work/$HOLDER/approaches/N5/sass_hist.awk" | grep -i "${3:-.}"
}
