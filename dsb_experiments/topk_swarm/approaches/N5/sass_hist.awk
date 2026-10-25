# Opcode histogram per kernel from cuobjdump -sass (after approaches/W1/w1x_sass_hist.awk, with the
# SIMD and ballot opcodes N5 cares about). Usage: cuobjdump -sass <bin> | awk -f sass_hist.awk
BEGIN {
  split("SHFL HMNMX2 VHMNMX HSET2 VIMNMX VIMNMX3 IMNMX ISETP PLOP3 SEL LOP3 IADD3 IMAD SHF PRMT POPC VOTE STS LDS LDG P2R R2P BSSY BSYNC BRA STL LDL WARPSYNC", a, " ")
  na = 29
}
/Function :/ { fn = $3 }
/^[ \t]*\/\*[0-9a-f]+\*\// {
  n[fn]++
  line = $0
  sub(/\/\*[0-9a-f]+\*\//, "", line)
  if (line ~ /^[ \t]*@/) { pred[fn]++ }
  sub(/^[ \t]*@!?U?P[0-9T][ \t]+/, "", line)
  sub(/^[ \t]*/, "", line)
  split(line, w, /[ .;]/)
  op = w[1]
  if (op ~ /^\{/) { sub(/^\{[ \t]*/, "", line); split(line, w, /[ .;]/); op = w[1] }
  c[fn, op]++
}
END {
  for (f in n) {
    printf "%s n=%d pred=%d", f, n[f], pred[f]
    for (i = 1; i <= na; i++) if (c[f, a[i]]) printf " %s=%d", a[i], c[f, a[i]]
    printf "\n"
  }
}
