#!/usr/bin/env bash
# Periodic broker tick: reconcile, release nearly dead unheld nodes, renew to the target,
# bootstrap new nodes. Runs until the given UTC time, then exits.
#   tools/broker_loop.sh <target> <until-utc-iso> [interval-seconds]
# <target> is a total such as 4, or a per-arch spec such as b200=3,gb200=1.
# Log: state/logs/tick.log (one block per tick, read the tail).
set -u
TARGET=${1:?target}
UNTIL=${2:?until, e.g. 2026-10-25T14:30:00Z}
INTERVAL=${3:-300}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$HERE/../state/logs/tick.log"
mkdir -p "$(dirname "$LOG")"
until_s=$(python3 -c "import datetime,sys; print(int(datetime.datetime.strptime(sys.argv[1],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()))" "$UNTIL")
echo "=== loop start $(date -u +%Y-%m-%dT%H:%M:%SZ) target=$TARGET until=$UNTIL interval=${INTERVAL}s pid=$$" >> "$LOG"
while [ "$(date -u +%s)" -lt "$until_s" ]; do
  {
    echo "=== tick $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    case "$TARGET" in
      *=*) python3 "$HERE/broker.py" tick --per-arch "$TARGET" --wait 10m 2>&1 | tail -n 40 ;;
      *)   python3 "$HERE/broker.py" tick --target "$TARGET" --wait 10m 2>&1 | tail -n 40 ;;
    esac
  } >> "$LOG" 2>&1
  sleep "$INTERVAL"
done
echo "=== loop end $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
