# Opcode histogram per kernel from cuobjdump -sass. Usage: cuobjdump -sass <bin> | awk -f w1x_sass_hist.awk
BEGIN {
  split("SHFL VIMNMX VIMNMX3 IMNMX ISETP FSETP PLOP3 SEL FSEL LOP3 IADD3 SHF PRMT P2R R2P BSSY BSYNC BRA STL LDL VOTE", a, " ")
  na = 21
}
/Function :/ { fn = $3 }
/^[ \t]*\/\*[0-9a-f]+\*\// {
  n[fn]++
  line = $0
  sub(/\/\*[0-9a-f]+\*\//, "", line)
  sub(/^[ \t]*@!?U?P[0-9T][ \t]+/, "", line)
  sub(/^[ \t]*/, "", line)
  split(line, w, /[ .;]/)
  op = w[1]
  if (op ~ /^\{/) { sub(/^\{[ \t]*/, "", line); split(line, w, /[ .;]/); op = w[1] }
  c[fn, op]++
}
END {
  for (f in n) {
    printf "%s n=%d", f, n[f]
    for (i = 1; i <= na; i++) if (c[f, a[i]]) printf " %s=%d", a[i], c[f, a[i]]
    printf "\n"
  }
}
