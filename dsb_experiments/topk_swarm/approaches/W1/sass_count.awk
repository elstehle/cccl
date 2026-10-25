BEGIN { split("SHFL VIMNMX IMNMX ISETP SEL FSETP FSEL FMNMX LOP3 PRMT BRA STL LDL PLOP3 IADD3 SHF VOTE POPC", a, " "); for (i in a) ops[a[i]] = 1 }
/Function :/ { name = $3; sub(/^_Z[0-9]+/, "", name); fn = substr(name, 1, 60) }
/^[ \t]*\/\*[0-9a-f]+\*\// { n[fn]++; for (op in ops) if ($0 ~ ("[ \t@!P0-9]" op "[ .]")) c[fn, op]++ }
END { for (f in n) if (f ~ /lat_kernel|thr_kernel/) { printf "%-62s n=%4d", f, n[f]; for (i = 1; i <= 18; i++) if (c[f, a[i]]) printf " %s=%d", a[i], c[f, a[i]]; printf "\n" } }
