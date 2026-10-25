# Node workflow for N10 (holder opt-N10). Source, then call:
#   n10_target <node>   n10_ship   n10_build <label> <run.py args>   n10_measure <label> <run.py args>   n10_fetch
# The harness shipped is the one committed in the N10 worktree (harness 0.4 plus the N10 adapters),
# SOURCE_COMMIT is written from the worktree HEAD before every ship. Logs stay on the node under
# <root>/work/opt-N10/, only the exit code and the last lines come back.
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm
WT=/Users/estehle/Downloads/cccl-topk-swarm-N10
N10=$WT/dsb_experiments/topk_swarm
HOLDER=opt-N10
STATE=$SWARM/state/$HOLDER
mkdir -p $STATE $SWARM/state/results/$HOLDER

n10_target() {
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
  n10_env
}
n10_env() { source $STATE/target.env; }
n10_ssh() {
  n10_env
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC_PREFIX bash -c $(printf %q "$1")"
}
n10_ship() {
  n10_env
  if [ -n "$(git -C $WT status --porcelain -- dsb_experiments/topk_swarm/harness dsb_experiments/topk_swarm/approaches)" ]; then
    echo "worktree has uncommitted harness or approach changes, commit first"; return 1
  fi
  git -C $WT rev-parse HEAD > $N10/harness/SOURCE_COMMIT
  n10_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER"
  tar czf - -C $N10 harness approaches/W1 approaches/N10 | n10_ssh "tar xzf - -C $ROOT/work/$HOLDER" || return 1
  echo "shipped $(cat $N10/harness/SOURCE_COMMIT)"
}
n10_runpy() {
  n10_env
  local log=$1; shift
  n10_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu 0 --node-name $NODE_NAME --arch-name $ARCH --source-commit $(git -C $WT rev-parse --short=10 HEAD) $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; grep -c compile_error $ROOT/work/$HOLDER/$log.log; tail -n 4 $ROOT/work/$HOLDER/$log.log | cut -c1-220"
}
n10_build() { n10_runpy build_$1 --build-only "${@:2}"; }
n10_measure() { n10_runpy measure_$1 --measure-only "${@:2}"; }
n10_fetch() {
  n10_env
  n10_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -12
}
# background form: start run.py under nohup on the node, poll with n10_poll <label>
n10_bg() {
  n10_env
  local log=$1; shift
  n10_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && nohup bash -c 'python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu 0 --node-name $NODE_NAME --arch-name $ARCH --source-commit $(git -C $WT rev-parse --short=10 HEAD) $*; echo exit=\$?' > $ROOT/work/$HOLDER/$log.log 2>&1 < /dev/null & echo started $log"
}
n10_poll() {
  n10_env
  local log=$1
  n10_ssh "cd $ROOT/work/$HOLDER && if grep -q '^exit=' $log.log; then echo DONE; grep -c compile_error $log.log; tail -n 5 $log.log | cut -c1-240; else echo RUNNING; tail -n 2 $log.log | cut -c1-200; fi"
}
# ship a local script and start it detached on the node (stdin does not pass through docker exec here, so base64)
n10_batch() {
  n10_env
  local script=$1 name=$(basename $1 .sh)
  local b64=$(base64 < $script | tr -d '\n')
  n10_ssh "cd $ROOT/work/$HOLDER && echo $b64 | base64 -d > $name.sh && chmod +x $name.sh && (setsid nohup ./$name.sh > $name.log 2>&1 < /dev/null &) ; sleep 2; head -3 $name.log"
}
n10_batchlog() { n10_env; n10_ssh "cat $ROOT/work/$HOLDER/$1.log"; }
