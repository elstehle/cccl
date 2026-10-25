# Node workflow for the W1x codegen experiment (holder exp-W1codegen). Source, then call.
#   w1x_target <node>   w1x_ship   w1x_build <run.py args>   w1x_measure <run.py args>   w1x_fetch
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
WX=/Users/estehle/Downloads/cccl-topk-swarm-W1x/dsb_experiments/topk_swarm
HOLDER=exp-W1codegen
STATE=$SWARM/state/$HOLDER
mkdir -p $STATE $SWARM/state/results/$HOLDER

w1x_target() {
  python3 - "$1" > $STATE/target.env <<'PY'
import json, sys, shlex
reg = json.load(open("/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm/state/nodes.json"))
name = sys.argv[1]
for n in reg["nodes"]:
    if (n["ssh"] == name or n["name"] == name) and n.get("status") == "ready":
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
w1x_env() { source $STATE/target.env; }
w1x_ssh() {
  w1x_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}
w1x_ship() {
  w1x_env
  w1x_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $SWARM harness | w1x_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  tar czf - -C $WX harness/adapters approaches/W1 | w1x_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo shipped
}
w1x_runpy() {
  w1x_env
  local log=$1; shift
  w1x_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu 0 --node-name $NODE_NAME --arch-name $ARCH --source-commit $(git -C $WX rev-parse --short=10 HEAD) $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; tail -n 3 $ROOT/work/$HOLDER/$log.log | cut -c1-200"
}
w1x_build() { w1x_runpy build_$1 --build-only "${@:2}"; }
w1x_measure() { w1x_runpy measure_$1 --measure-only "${@:2}"; }
w1x_fetch() {
  w1x_env
  w1x_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -8
}
