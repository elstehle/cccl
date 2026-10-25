# opt-W1b node helpers. Source, then: w1b_runpy <log> <run.py args>  (build or measure)
# Logs stay on the node under <root>/work/opt-W1b/, only the exit code and the last lines come back.
NODE=${NODE:-umbriel-b200-091}
NODE_NAME=${NODE_NAME:-umbriel-b200-091}
ARCH=${ARCH:-b200}
ROOT=${ROOT:-/work/topk_swarm}
EXEC=${EXEC:-"docker exec -i topk_swarm"}
HOLDER=opt-W1b
W1B=/Users/estehle/Downloads/cccl-topk-swarm-W1b/dsb_experiments/topk_swarm
SWARM=/Users/estehle/Downloads/cccl-topk-swarm/dsb_experiments/topk_swarm

w1b_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$NODE" "$EXEC bash -c $(printf %q "$1")"; }
w1b_ship() { (cd $W1B && tar czf - harness approaches/W1) | w1b_ssh "mkdir -p $ROOT/work/$HOLDER $ROOT/results/$HOLDER && tar xzf - -C $ROOT/work/$HOLDER" && echo shipped; }
w1b_runpy() {
  local log=$1; shift
  w1b_ssh "cd $ROOT/work/$HOLDER/harness && export PATH=/usr/local/cuda/bin:\$PATH && python3 run.py --repo $ROOT/cccl --root $ROOT --holder $HOLDER --gpu 0 --node-name $NODE_NAME --arch-name $ARCH $* > $ROOT/work/$HOLDER/$log.log 2>&1; echo exit=\$?; grep -c compile_error $ROOT/work/$HOLDER/$log.log; tail -n 4 $ROOT/work/$HOLDER/$log.log | cut -c1-220"
}
w1b_fetch() { w1b_ssh "tar czf - -C $ROOT/results/$HOLDER ." | tar xzf - -C $SWARM/state/results/$HOLDER && ls -t $SWARM/state/results/$HOLDER | head -8; }
