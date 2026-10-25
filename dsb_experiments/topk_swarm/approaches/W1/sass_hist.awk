# Opcode histogram of the latency and throughput kernels in a cuobjdump -sass dump.
#   cuobjdump -sass <binary> | awk -f sass_hist.awk
/Function :/ { fn = $3; sub(/^_Z[0-9]+/, "", fn); fn = substr(fn, 1, 70) }
/^[ \t]*\/\*[0-9a-f]+\*\// {
  n[fn]++
  line = $0; sub(/^[ \t]*\/\*[0-9a-f]+\*\/[ \t]*/, "", line); sub(/^@!?U?P[0-9T] +/, "", line)
  op = line; sub(/ .*/, "", op); sub(/;.*/, "", op); sub(/\..*/, "", op)
  c[fn, op]++; ops[op] = 1
}
END {
  for (f in n) if (f ~ /lat_kernel|thr_kernel/) {
    printf "%-72s n=%5d", f, n[f]
    for (op in ops) if (c[f, op] >= 8) printf " %s=%d", op, c[f, op]
    printf "\n"
  }
}
