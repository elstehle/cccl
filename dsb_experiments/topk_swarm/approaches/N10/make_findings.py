#!/usr/bin/env python3
"""Write the findings JSON files for the N10 cards (one adapter per card). Shared contract, per-variant
findings. Run from approaches/N10, then tools/make_card.py with the batches listed in REPORT.md."""
import json, sys

CONTRACT = {
  "input_location": "registers",
  "input_arrangement": "striped",
  "output_location": "smem",
  "output_form": "indices",
  "output_sorted": False,
  "temp_storage_bytes": 0,
  "sync_assumed": "all lanes of the hardware warp call the primitive together (the logical warps of a sub-warp instantiation run in lockstep)",
  "sync_performed": "shuffles, ballots and one redux on the full warp mask, no barrier",
  "k_range": "[1, N] with MaxK = max(LW, next_pow2(k)) dividing N and N / MaxK a power of two, the truncated merge applies to a power of two k (other k take the complete merge and a shuffle), k 4 to 64 measured",
  "n_range": "N = LW x IPT with MaxK dividing N and N / MaxK a power of two, measured 32 to 256 at LW 32 and 8 to 128 at LW 4, 8, 16",
  "type_constraints": "fp32 keys only, twiddled to an order-preserving u32 word (sign flip, -0.0 mapped to +0.0), the index is never carried through the network and is recovered from the item's position (slot i of lane l is source index i x LW + l) by ballot rank",
  "nan_behavior": "a NaN twiddles above +inf (positive sign) or below -inf (negative sign) and is selected or dropped deterministically by its sign bit, never produced by focus-set inputs. -0.0 and +0.0 are one word",
  "tie_break_support": "prefer_smaller_index exact by construction (items above the k-th word in index order, then the tied items in index order until k), the same code serves not_guaranteed at the same cost (within 0.04 percent in every measured pair of cells)",
  "sorted_output_support": "not provided: the output is the selected set in class and index order (the complete-merge form N10_u32 has the sorted keys in position but places indices by rank, not by position)"
}
COMMON_SRC = ["approaches/N10/n10_detail.cuh", "approaches/W1/warp_level_topk.cuh", "approaches/W1/w1_common.cuh"]
ADOPTED = ["T1", "T4", "T7", "T17", "T28", "T34", "T36", "T37", "T40", "T42"]

def write(variant, adapter, findings, notes, contributed=()):
    d = {
      "author": "opt-N10", "approach_id": "N10", "variant": variant,
      "contract": dict(CONTRACT),
      "report_path": "approaches/N10/REPORT.md",
      "source_paths": ["harness/adapters/%s.cuh" % adapter] + COMMON_SRC,
      "techniques_adopted": ADOPTED, "techniques_contributed": list(contributed),
      "correctness_notes": notes, "findings": findings,
    }
    for k in ("low_level", "algorithmic", "negative", "surprises", "open_questions"):
        assert findings.get(k), (variant, k)
    json.dump(d, open("findings_%s_b200.json" % variant, "w"), indent=1)
    print("wrote findings_%s_b200.json" % variant)

if __name__ == "__main__":
    exec(open(sys.argv[1]).read())
