# Techniques catalogue (append only)

Generally applicable optimizations, intrinsic costs and codegen lessons, one entry each, appended with `tools/append_locked.py`. Every number carries arch, CUDA version, node, date and the microbenchmark file. The knowledge curator folds entries into `TOPK_KNOWLEDGE_BASE.md` at every wave boundary. Entry ids are `T<n>`, assigned in order of appending.

Entry format:

```
### T<n>: <name>
applies to: <scopes, approaches, key widths>
measured: <number, arch, CUDA, node, date, file>
evidence: <SASS lines, file, or report section>
how to use: <one paragraph>
caveats: <where it does not hold>
author: <agent>
```

Seed entries from the July 2026 study are in `reference/BLOCK_TOPK_RESULTS.md` section 1 and section 6 (B200, CUDA 13.1 and 13.3.1, nodes umb-b200-2xx). The hardware researcher re-measures them on the current toolchain before adding them here as T entries, so that no stale number enters the catalogue.

---
