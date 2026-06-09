#!/usr/bin/env bash
# Convenience driver: build, run the latency sweep, and run the per-kernel breakdown.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/build.sh"
"$SCRIPT_DIR/run_sweep.sh"
"$SCRIPT_DIR/run_breakdown.sh"
