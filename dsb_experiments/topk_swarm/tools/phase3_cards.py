#!/usr/bin/env python3
"""Phase 3 cards: one card per winner or named runner-up adapter, built from the phase 3 batches of
one arch with make_card.py, the contract from the adapter header and the batch statics, and the
findings written from the winners reduction (wins, losses, resources, the caveats the study found).

  python3 tools/phase3_cards.py --arch b200 [--adapters B4_v4,...] [--dry-run]

Findings files go to state/phase3/findings/<adapter>_<arch>.json, cards to cards/<approach>_<variant>_<arch>_<yyyymmdd>_1.json.
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SWARM))

# adapter id -> (approach_id, variant, labels in the winners reduction, report path)
ADAPTERS = {
    "B4_v4": ("B4", "v4", ["B4_v4", "B4_v4@T512", "B4_v4@T128"], "approaches/B4/REPORT.md"),
    "B2_packed": ("B2", "packed", ["B2_packed", "B2_packed@T512"], "approaches/B2/REPORT.md"),
    "N8_b2packed": ("B2", "packed_t128", ["B2_packed@T128"], "approaches/N8/REPORT.md"),
    "N9_w2fin2": ("N9", "w2fin2", ["N9_w2fin2", "N9_w2fin2@T512", "N9_w2fin2@T128"], "approaches/N9/REPORT.md"),
    "B3_tuned": ("B3", "tuned", ["B3_tuned"], "approaches/B3/REPORT.md"),
    "B1_det2": ("B1", "det2", ["B1_det2", "B1_det2@T512", "B1_det2@T128"], "approaches/B1/REPORT.md"),
    "B1_det3": ("B1", "det3", ["B1_det3", "B1_det3@T512", "B1_det3@T128"], "approaches/B1/REPORT.md"),
    "B1_n6b": ("B1", "n6b", ["B1_n6b", "B1_n6b@T512", "B1_n6b@T128"], "approaches/B1/REPORT.md"),
    "B1_idx": ("B1", "idx", ["B1_idx", "B1_idx@T512", "B1_idx@T128"], "approaches/B1/REPORT.md"),
    "W2_packed_v3": ("W2", "packed_v3", ["W2_packed_v3"], "approaches/W2/REPORT.md"),
    "W5_v4": ("W5", "v4", ["W5_v4"], "approaches/W5/REPORT.md"),
    "N6w_atom32": ("N6", "w_atom32", ["N6w_atom32"], "approaches/N6w/REPORT.md"),
    "N5_rows_f": ("N5", "rows_f", ["N5_rows_f"], "approaches/N5/REPORT.md"),
    "W3_d4c": ("W3", "d4c", ["W3_d4c"], "approaches/W3/REPORT.md"),
    "N10_auto": ("N10", "auto", ["N10_auto"], "approaches/N10/REPORT.md"),
    "W1_level": ("W1", "level", ["W1_level"], "approaches/W1/REPORT.md"),
    "W1x_vimnmx_eager": ("W1", "x_vimnmx_eager", ["W1x_vimnmx_eager"], "approaches/W1/REPORT.md"),
    "W4_chunk4": ("W4", "chunk4", ["W4_chunk4@runs4"], "approaches/W4/REPORT.md"),
    "W4_wmerge": ("W4", "wmerge", ["W4_wmerge"], "approaches/W4/REPORT.md"),
    "W2_packed_sw_v2": ("W2", "packed_sw_v2", ["W2_packed_sw_v2@T4", "W2_packed_sw_v2@T8", "W2_packed_sw_v2@T16"], "approaches/W2/REPORT.md"),
    "W1_eager_same_sub": ("W1", "eager_same_sub", ["W1_eager_same_sub@T4", "W1_eager_same_sub@T8", "W1_eager_same_sub@T16"], "approaches/W1/REPORT.md"),
    "N10_auto_sub": ("N10", "auto_sub", ["N10_auto_sub@T4", "N10_auto_sub@T8", "N10_auto_sub@T16"], "approaches/N10/REPORT.md"),
    "N5_rows_fs_sub": ("N5", "rows_fs_sub", ["N5_rows_fs_sub@T4", "N5_rows_fs_sub@T8", "N5_rows_fs_sub@T16"], "approaches/N5/REPORT.md"),
}

# caveats the study established, by arch and adapter: negative, surprises, open_questions, the sorted
# verdict (True, False or None for "derive from the data"), optional sorted_support text, optional
# correctness note, optional low_level lines, unsorted_by_contract for adapters whose output is unsorted by design
CAVEATS_B200 = {
    "W5_v4": {
        "negative": ["Phase 3b (umbriel-b200-047): six fp32 prefer_smaller_index cells fail the harness sorted-order check, k 16 and 64 at N 512, 2048 and 4096, on the quantized_random and relu_quantized rows (the signed-zero and tie rows passed). The sorted finish leaves equal keys inside the top k in scatter position order because the tie rewrite fixes only the class at the k-th slot. The unsorted cells stand."],
        "surprises": ["On tie-heavy input the threshold filter loses its edge: the warp throughput cells it wins on random input go to W2_packed_v3 and N6w_atom32 under tie_heavy."],
        "open_questions": ["The fix is the tie rewrite generalized to every class run inside the top k, open is whether the sorted output then costs anything over the unsorted path."],
        "sorted": False,
        "sorted_support": "sorted finish present, fails the phase 3b order check for fp32 ties (see the negative findings)",
        "correctness": "the oracle failures are the phase 3b sorted-order check under prefer_smaller_index for fp32 (k 16 and 64 at N 512, 2048 and 4096, quantized_random and relu_quantized rows, equal keys inside the top k left in scatter order), a real ordering defect of the sorted finish, every unsorted cell passed the multiset and index-set checks"},
    "N5_rows_f": {
        "negative": ["No sorted output by contract (card: sorted_output_support none): every phase 3b cell fails the order check by design and is recorded as not applicable, the unsorted cells stand. Throughput only by design, the two-row-per-lane form has no latency reading."],
        "surprises": [],
        "open_questions": ["A sorted epilogue for the two-row form, if any workload needs sorted output at throughput."],
        "sorted": False, "unsorted_by_contract": True, "sorted_support": "none, unsorted output by contract"},
    "N5_rows_fs_sub": {
        "negative": ["No sorted output by contract (card: sorted_output_support none), every phase 3b cell fails the order check by design and is recorded as not applicable, the unsorted cells stand. Throughput only."],
        "surprises": ["On GB200 the 16-lane form at N 128 runs lower in throughput than on B200 (bf16 k 16: 604 against 418 Gelem/s), which moves the cell to W2_packed_sw_v2 at 8 lanes, a generation effect confirmed on a second GB200 host."],
        "open_questions": [],
        "sorted": False, "unsorted_by_contract": True, "sorted_support": "none, unsorted output by contract"},
    "B4_v4": {
        "negative": ["On the unshuffled tie_heavy pattern (period 64 divides the 256-thread stripe, every striped thread holds 64 copies of one value) the minimum of the thread maxima equals the tile minimum, every item is a candidate and the tile fallback runs: fp32 N 16384 k 32 reads 19153 cycles against 4626 on random input, behind B3_tuned. The shuffled tie pattern (harness 0.11) removes the alignment and returns B4_v4 to its fast path (5316, ahead of B3_tuned at 18582)."],
        "surprises": ["The phase 3 region N 8192 to 32768 runs at 254 to 255 registers, one block per SM, so the win is a latency win at low occupancy. verify-B4 confirmed the phase 3 card cells on umbriel-b200-068 under harness 0.10 (48 cells, ratios 0.965 to 1.043, no flips), the wave 2 cards had been confirmed under harness 0.8, and the k 4096 cells stay behind B3_tuned."],
        "open_questions": ["A cheaper fallback detection for tie floods (count the bound's ties before running the full tile path) and whether k = N/4 can be rescued by a second bound."],
        "sorted": None},
    "B1_det3": {
        "negative": ["det2 and det3 share a node only on tie_heavy input (umbriel-b200-068), where det3 is 1.3 to 5.3 percent behind B1_det2 at N 1024 to 8192. On random input the comparison is cross-node (det3 in the 068 addendum, det2 in the 047 pass): within 2 percent in latency at N 1024 and up to 15 percent at N 16384. det3 beats no umbriel-b200-047 winner (state/phase3/det3_cells_b200.md): the deterministic price is paid, not removed."],
        "surprises": ["The addendum was measured on umbriel-b200-068 while the random pass ran on umbriel-b200-047, so its random-input rows sit in a node-named addendum table and the comparison with the 047 winners is cross-node. The same-node rerun on 047 never got its node (follow-up in PHASE3_PLAN.md 6.3)."],
        "open_questions": ["Same-node rerun against B1_det2 and B1_idx on the random-pass node, and the bf16 deterministic price at N 8192 and above."],
        "sorted": None},
    "B1_det2": {"negative": [], "surprises": ["In throughput B1_det2 and B1_idx tie within 0.1 percent on most N 1024 cells (the deterministic rank costs nothing there), the matrix lists whichever was marginally faster."], "open_questions": [], "sorted": None},
    "B2_packed": {
        "negative": ["Above 32 items per thread (N 16384 and 32768 at 256 threads) the packed list spills to local memory (T34), so its margins there are those of a degraded form and the matrix marks it with (spills) as a runner-up."],
        "surprises": [], "open_questions": ["A two-level list or a smaller per-lane window for N above 8192."], "sorted": None},
    "N8_b2packed": {
        "negative": ["The same kernel as B2_packed launched at 128 threads (the matrices report it as B2_packed@T128): its win region is small k at N 1024 to 8192, with the B1_reference control also instantiated at 128 threads in its cells."],
        "surprises": ["At fp32 N 8192 k 16 the 128-thread form wins at 254 registers, a cell the N8 card recommended 256 threads for."],
        "open_questions": ["Where the 128 against 256 thread crossover sits per key type when both are measured against the same control."], "sorted": None},
    "B1_n6b": {
        "negative": ["The equi-width first pass gains its exact-count exit from the aligned tie_heavy pattern as much as B4_v4 loses from it: fp32 N 8192 k 32 reads 4499 on random, 9306 on tie_heavy and 7378 on shuffled ties."],
        "surprises": [], "open_questions": ["The bracket choice for bf16 at N 16384 and above where it trails B4_v4 by 2x on random input."], "sorted": None},
    "W4_chunk4": {
        "negative": ["Runs only on its sorted-runs input (4 presorted runs), so it is a separate track in the matrix with B1_reference on the same input as its control and B1_det3@runs4 as a reference column, never head to head with the random-input winners.",
                     "The 48 oracle failures recorded against W4_chunk4 in the phase 3 block pass belong to the quarantined batches that fed 8 runs to the 4-run adapter (state/results/eval-engineer/phase3/b200/rejected_contract_mismatch), they are not part of this card."],
        "surprises": [], "open_questions": ["The merge against B3_tuned and B4_v4 on the same presorted input, which the sorted-runs track now allows."], "sorted": True},
    "W4_wmerge": {"negative": ["fp32 N 2048 cells run in the latency regime only: eight warps of its temp storage exceed the dynamic shared memory limit in the throughput block, which supported() cannot express."], "surprises": [], "open_questions": [], "sorted": True},
    "W2_packed_v3": {"negative": ["Not sorted for fp32: the N4 truncated-key route leaves keys that are equal under truncation in index order (verify-W4), bf16 passes the order check."], "surprises": [], "open_questions": ["Whether the fp32 route can carry the full key for the last compare without losing the packed round."], "sorted": None},
    "W2_packed_sw_v2": {"negative": [],
                        "surprises": ["The named sub-warp workloads pick the 16-lane form on 4 of 6 cells once every lane count is measured (the first pass ran them at 4 lanes only).",
                                      "On B200 the named-workload rows at 8 and 16 lanes come from the umbriel-b200-068 addendum (the 047 pass ran the named sub-warp cells at 4 lanes only), so the lane-count choice there is a cross-node comparison, and the single-host evidence is the GB200 pass on gb200-nvl4-ts2-99."],
                        "open_questions": [], "sorted": None},
}

GB200_SCOPE = "The GB200 pass measured random input only on one host (gb200-nvl4-ts2-99): the tie_heavy, shuffled-tie and phase 3b order-check findings for this adapter are on its B200 card."
# adapters whose output is sorted by construction and passed the B200 phase 3b order check
SORTED_ON_B200 = {"W1_level", "W1x_vimnmx_eager", "W1_eager_same_sub", "W4_wmerge", "W4_chunk4"}
CAVEATS_GB200 = {
    "W5_v4": {
        "negative": [GB200_SCOPE, "The fp32 sorted-finish defect found by the B200 phase 3b order check (equal keys inside the top k left in scatter order under prefer_smaller_index, k 16 and 64 at N 512, 2048 and 4096) is a property of the same code and stands for this card, the GB200 pass measured unsorted output only."],
        "surprises": [], "open_questions": ["The fix is the tie rewrite generalized to every class run inside the top k, open is whether the sorted output then costs anything over the unsorted path."],
        "sorted": False, "sorted_support": "sorted finish present, fails the B200 phase 3b order check for fp32 ties, not rerun on GB200"},
    "N5_rows_f": {
        "negative": ["No sorted output by contract (card: sorted_output_support none). Throughput only by design, the two-row-per-lane form has no latency reading.", GB200_SCOPE],
        "surprises": [], "open_questions": ["A sorted epilogue for the two-row form, if any workload needs sorted output at throughput."],
        "sorted": False, "unsorted_by_contract": True, "sorted_support": "none, unsorted output by contract"},
    "N5_rows_fs_sub": {
        "negative": ["No sorted output by contract (card: sorted_output_support none). Throughput only.", GB200_SCOPE],
        "surprises": ["At N 128 the 16-lane form runs lower in throughput than on B200 (bf16 k 16: 418 against 604 Gelem/s), which moves the cell to W2_packed_sw_v2 at 8 lanes (519 Gelem/s), a generation effect confirmed on a second GB200 host."],
        "open_questions": [], "sorted": False, "unsorted_by_contract": True, "sorted_support": "none, unsorted output by contract"},
    "B4_v4": {
        "negative": [GB200_SCOPE + " On B200 the aligned tie_heavy pattern sends B4_v4 to its tile fallback, a code property that holds on this host too."],
        "surprises": [], "open_questions": ["A cheaper fallback detection for tie floods (count the bound's ties before running the full tile path) and whether k = N/4 can be rescued by a second bound."],
        "sorted": None},
    "B1_det3": {
        "negative": [GB200_SCOPE],
        "surprises": ["det3 and det2 ran in the same GB200 pass on one host: over the 39 random cells where both have a value det3 is between 5 percent ahead and 2 percent behind B1_det2 in latency, and ahead in throughput at N 8192 for fp32 k 1024 and 2048 under prefer_smaller_index (510 to 541 against 385 to 392 Gelem/s), the cells where it is the winner."],
        "open_questions": ["The bf16 deterministic price at N 8192 and above."], "sorted": None},
    "B1_det2": {"negative": [GB200_SCOPE], "surprises": ["In throughput B1_det2 and B1_idx tie within 0.5 percent on all 47 random cells where both have a value, the matrix lists whichever was marginally faster."], "open_questions": [], "sorted": None},
    "B2_packed": {
        "negative": ["Above 32 items per thread (N 16384 and above at 256 threads, outside the GB200 pass) the packed list spills to local memory on B200 (T34), the GB200 cells up to N 8192 run without spills.", GB200_SCOPE],
        "surprises": [], "open_questions": ["A two-level list or a smaller per-lane window for N above 8192."], "sorted": None},
    "N8_b2packed": {
        "negative": ["The same kernel as B2_packed launched at 128 threads (the matrices report it as B2_packed@T128), with the B1_reference control also instantiated at 128 threads in its cells.", GB200_SCOPE],
        "surprises": [], "open_questions": ["Where the 128 against 256 thread crossover sits per key type when both are measured against the same control."], "sorted": None},
    "B1_n6b": {"negative": [GB200_SCOPE + " On B200 the aligned tie pattern gives the equi-width first pass an exact-count exit that the shuffled pattern removes."], "surprises": [], "open_questions": [], "sorted": None},
    "W4_chunk4": {
        "negative": ["Runs only on its sorted-runs input (4 presorted runs), so it is a separate track in the matrix with B1_reference on the same input as its control, never head to head with the random-input winners.", GB200_SCOPE],
        "surprises": [], "open_questions": ["The merge against B3_tuned and B4_v4 on the same presorted input, which the sorted-runs track now allows."], "sorted": True},
    "W4_wmerge": {"negative": [GB200_SCOPE], "surprises": [], "open_questions": [], "sorted": True},
    "W2_packed_v3": {"negative": ["Not sorted for fp32 (B200 phase 3b: the N4 truncated-key route leaves keys that are equal under truncation in index order), bf16 passes the order check there.", GB200_SCOPE], "surprises": [], "open_questions": ["Whether the fp32 route can carry the full key for the last compare without losing the packed round."], "sorted": None},
    "W2_packed_sw_v2": {"negative": [GB200_SCOPE], "surprises": ["Every lane count of the named sub-warp cells was measured on the one GB200 host, the single-host evidence for the lane-count choice that B200 lacks (its 8 and 16 lane named rows are a cross-node addendum)."], "open_questions": [], "sorted": None},
}

GENERIC_OPEN_B200 = "GB200 pass on gb200-nvl4-ts2-99 (WINNERS_gb200.md): the warp and sub-warp winners agree on 359 of 376 common cells (365 of 376 when each cell is compared through the adapters both passes measured, as section 16 of the knowledge base does), the differences are named sub-warp lane-count cells, N 128 sub-warp throughput cells and warp ties within 5 percent, and in the block matrices the B1_idx against B1_det2 throughput ties fall either way. The GB200 k 64 cells at 128 threads run 3 to 15 percent slower than on B200 (hw-researcher, a generation property)."
GENERIC_OPEN_GB200 = "B200 pass (WINNERS_b200.md, random input on umbriel-b200-047, tie_heavy on 068, shuffled ties on 094, phase 3b on 047): the warp and sub-warp winners agree with GB200 on 359 of 376 common cells (365 of 376 when each cell is compared through the adapters both passes measured, as section 16 of the knowledge base does), the differences are named sub-warp lane-count cells, N 128 sub-warp throughput cells and warp ties within 5 percent, and in the block matrices the B1_idx against B1_det2 throughput ties fall either way. On GB200 the k 64 cells at 128 threads run 3 to 15 percent slower than on B200 (hw-researcher, a generation property) and the aarch64 cicc emits branches for the conditional swaps (T20, 3 to 4 percent)."


def caveats_for(arch, aid):
    table = CAVEATS_B200 if arch == "b200" else CAVEATS_GB200
    cav = dict(table.get(aid, {}))
    cav.setdefault("negative", [] if arch == "b200" else [GB200_SCOPE])
    cav.setdefault("surprises", [])
    cav.setdefault("open_questions", [])
    if arch != "b200":
        cav.setdefault("sorted", True if aid in SORTED_ON_B200 else None)
        if "sorted_support" not in cav:
            cav["sorted_support"] = ("sorted by construction, order check passed on B200 in phase 3b, not rerun on GB200"
                                     if aid in SORTED_ON_B200 else "not sorted (see the B200 card)")
    else:
        cav.setdefault("sorted", None)
    return cav



def sh(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()


def adapter_commit(aid):
    return sh(["git", "log", "-1", "--format=%h", "--", "dsb_experiments/topk_swarm/harness/adapters/%s.cuh" % aid], cwd=REPO) or "unknown"


def header_contract(aid):
    try:
        text = open(os.path.join(SWARM, "harness", "adapters", aid + ".cuh")).read()
    except OSError:
        return "", ""
    m = re.search(r'contract\s*=\s*((?:\s*"[^"]*")+)', text)
    contract = " ".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""
    sc = re.search(r"topk::scope::(\w+)", text)
    return contract, (sc.group(1) if sc else "")


def pct(x):
    return "%+.0f percent" % (x * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="b200")
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--date", default="20261026")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results = os.path.join(SWARM, "state", "results", "eval-engineer", "phase3", args.arch)
    winners = json.load(open(os.path.join(SWARM, "state", "phase3", "winners_%s.json" % args.arch)))["rows"]
    fdir = os.path.join(SWARM, "state", "phase3", "findings")
    os.makedirs(fdir, exist_ok=True)
    wanted = args.adapters.split(",") if args.adapters else list(ADAPTERS)
    written = []
    for aid in wanted:
        approach, variant, labels, report = ADAPTERS[aid]
        files = sorted(f for f in glob.glob(os.path.join(results, "%s_vs_*.json" % aid)) if not f.endswith(".build.json"))
        if not files:
            print("no batches for", aid)
            continue
        loaded = [(f, json.load(open(f))) for f in files]
        # the card is built from the phase 3 passes only (pre-flight and check batches carry other labels)
        loaded = [(f, b) for f, b in loaded if str(b.get("label", "")).startswith("phase3_")]
        files = [f for f, b in loaded]
        batches = [b for f, b in loaded]
        if not batches:
            print("no phase 3 batches for", aid)
            continue
        # --- standing from the winners reduction
        standing = defaultdict(lambda: {"cells": 0, "won": 0, "runner": 0, "behind": [], "won_by": defaultdict(int)})
        nodes, versions = set(), set()
        random_nodes = set()
        nodes_by_pattern = defaultdict(set)
        runs_track = set()
        for r in winners:
            mine = [e for e in r["all"] if e["adapter"] in labels]
            if not mine:
                continue
            # the sorted-runs track (W4 merges) is reported under its pattern with the runs suffix stripped
            pattern = r["pattern"]
            if "@runs" in pattern:
                pattern = pattern.split("@runs")[0]
                runs_track.add(r["pattern"].split("@runs")[1])
            key = (pattern, r["regime"], r["requirement"])
            st = standing[key]
            st["cells"] += 1
            nodes_by_pattern[pattern].add(r["node"])
            if pattern == "random":
                random_nodes.add(r["node"])
            if r["winner"] in labels:
                st["won"] += 1
                st["won_by"][r["winner"]] += 1
            elif r["runner_up"] in labels:
                st["runner"] += 1
            else:
                me = mine[0]["value"]
                gap = (me - r["value"]) / r["value"] if r["regime"] == "latency" else (r["value"] - me) / me
                st["behind"].append((gap, r["winner"], r["key_type"], r["n"], r["k"], r["regime"], r["requirement"]))
            nodes.add(r["node"])
        # --- resources and statics from the batches
        regs, spills, smem, hv = [], 0, [], set()
        cells_total, oracle_cases, unsupported, failed_oracle, order_na = 0, 0, 0, 0, 0
        ks, ns, keys, arr, outf, ts_bytes, rows_per = set(), set(), set(), set(), set(), [], 1
        cav = caveats_for(args.arch, aid)
        seen_kernels = set()
        s3_pass, s3_none, s3_failed, s3_nodes = 0, 0, 0, set()
        for b in batches:
            if b.get("cells"):
                hv.add(b.get("harness_version") or "unknown")
            is_3b = str(b.get("label", "")).endswith("_3b")
            if is_3b:
                s3_pass += len(b.get("cells", []))
                s3_none += sum(1 for c in b.get("cells", []) if c.get("control_missing"))
                s3_nodes.add((b.get("node") or {}).get("hostname", "unknown"))
            for c in b.get("cells", []):
                # a cell-list batch under the other requirement repeats the same kernel without a control,
                # the resource counts take every kernel once
                kkey = (b.get("label"), (b.get("node") or {}).get("hostname"), c["key_type"], c.get("threads"), c["n"], c["k"],
                        c["requirement"], c["regime"], c.get("input_pattern"), c.get("named_pattern"), c.get("runs"))
                if kkey not in seen_kernels:
                    seen_kernels.add(kkey)
                    regs.append(c["resources"].get("registers", 0))
                    if c["resources"].get("spill_bytes", 0) > 0:
                        spills += 1
                    smem.append(c["resources"].get("smem_bytes", 0))
                ks.add(c["k"]); ns.add(c["n"]); keys.add(c["key_type"])
                if c["regime"] == "latency":
                    oracle_cases += int((c.get("raw", {}).get("variant") or {}).get("oracle_cases", 0))
                st = c.get("raw", {}).get("variant_static") or {}
                if st:
                    arr.add(st.get("input_arrangement")); outf.add(st.get("output_form")); ts_bytes.append(st.get("temp_storage_bytes", 0)); rows_per = st.get("rows", 1)
            cells_total += len(b.get("cells", []))
            unsupported += len(b.get("unsupported", []))
            for f in b.get("failed", []):
                msgs = f.get("failures", [])
                if f["reason"] != "oracle_failed" or "exceeds the device limit" in " ".join(msgs):
                    continue
                order_only = bool(msgs) and all(("output not sorted" in m) or ("not in the tie-break index order" in m) for m in msgs)
                if cav.get("unsorted_by_contract") and order_only:
                    order_na += 1
                else:
                    failed_oracle += 1
                    if is_3b:
                        s3_failed += 1
        contract_text, scope = header_contract(aid)
        contract_text = " ".join((contract_text or "").split())
        sorted_out = cav["sorted"] if cav["sorted"] is not None else bool(s3_pass > 0 and s3_failed == 0 and order_na == 0)

        # --- findings text
        def wins_line(pattern):
            parts = []
            for regime in ("latency", "throughput"):
                for req in ("not_guaranteed", "prefer_smaller_index"):
                    st = standing.get((pattern, regime, req))
                    if st and st["cells"]:
                        split = ""
                        if len(st["won_by"]) > 1:
                            split = ", ".join("%d as %s" % (n, lab) for lab, n in sorted(st["won_by"].items(), key=lambda x: -x[1])) + ", "
                        parts.append("%s %s wins %d of %d cells (%srunner-up in %d)" % (regime, req, st["won"], st["cells"], split, st["runner"]))
            return ", ".join(parts) if parts else ""
        algorithmic = []
        rl = wins_line("random")
        if rl:
            if runs_track:
                algorithmic.append("Random input in the sorted-runs track (%s presorted runs, the track's only candidate, so a win is a lead over the B1_reference control on the same input, the one-node matrix on %s): %s." % (
                    ", ".join(sorted(runs_track)), ", ".join(sorted(n for n in random_nodes if n)), rl))
            else:
                algorithmic.append("Random input (the one-node matrix on %s): %s." % (", ".join(sorted(n for n in random_nodes if n)), rl))
        # where it wins on random: median margin over reference and runner-up
        wins_rows = [r for r in winners if r["pattern"].split("@runs")[0] == "random" and r["winner"] in labels]
        if wins_rows:
            mr = [r["margin_runner_up"] for r in wins_rows if r["margin_runner_up"] is not None]
            mf = [r["margin_reference"] for r in wins_rows if r["margin_reference"] is not None]
            cells_desc = sorted({(r["key_type"], r["n"]) for r in wins_rows})
            algorithmic.append("Where it wins on random input (%s), the median margin over the runner-up is %s and over the scope's reference control %s." % (
                ", ".join("%s N %d" % c for c in cells_desc[:8]) + (" and more" if len(cells_desc) > 8 else ""),
                pct(statistics.median(mr)) if mr else "n/a", pct(statistics.median(mf)) if mf else "n/a"))
        best_gap = {}
        for key, st in standing.items():
            if key[0] != "random":
                continue
            for g, w, kt, n, k, regime, req in st["behind"]:
                ck = (kt, n, k)
                if ck not in best_gap or g > best_gap[ck][0]:
                    best_gap[ck] = (g, w, regime, req)
        behind = sorted(((g, w, kt, n, k, regime, req) for (kt, n, k), (g, w, regime, req) in best_gap.items()), reverse=True)[:4]
        if behind:
            algorithmic.append("Largest random-input deficits (distinct cells, worst regime): " + ", ".join(
                "%s N %d k %d %s behind %s in %s %s" % (kt, n, k, pct(g), w, regime, req) for g, w, kt, n, k, regime, req in behind) + ".")
        tl = wins_line("tie_heavy")
        if tl:
            algorithmic.append("Tie-heavy input (unshuffled, %s): %s." % (", ".join(sorted(nodes_by_pattern["tie_heavy"])), tl))
        sl = wins_line("tie_heavy_shuffled")
        if sl:
            algorithmic.append("Shuffled tie-heavy input (harness 0.11, %s): %s." % (", ".join(sorted(nodes_by_pattern["tie_heavy_shuffled"])), sl))
        named = [r for r in winners if r["pattern"] == "random" and r["workload"] and r["winner"] in labels]
        if named:
            algorithmic.append("Named workloads won on random input: " + ", ".join(sorted({r["workload"] for r in named})) + ".")
        if not algorithmic:
            algorithmic.append("Measured in the phase 3 passes without a winning cell in the reduction, see the cells for its standing against the control.")
        low_level = []
        if regs:
            low_level.append("Registers %d to %d over the %d distinct kernels behind the card's cells (%d with register spills, a kernel measured in more than one batch counts once), shared memory %d to %d bytes, harness %s on %s." % (
                min(regs), max(regs), len(regs), spills, min(smem), max(smem), ", ".join(sorted(hv)), ", ".join(sorted(n for n in nodes if n))))
        if ts_bytes:
            low_level.append("Adapter statics: input arrangement %s, output %s, temp storage %d to %d bytes%s." % (
                ", ".join(sorted(a for a in arr if a)), ", ".join(sorted(o for o in outf if o)), min(ts_bytes), max(ts_bytes),
                (", %d rows per group" % rows_per) if rows_per and rows_per > 1 else ""))
        oracle_parts = ["Oracle battery (23 cases per cell plus the measurement seeds) passed on every cell that has a number"]
        oracle_parts.append(("%d cells unsupported by the adapter's contract or the device" % unsupported) if unsupported else "no cell unsupported")
        oracle_parts.append(("%d cells failed the oracle and carry no number" % failed_oracle) if failed_oracle else "no cell failed the oracle")
        if order_na:
            oracle_parts.append("%d phase 3b cells failed only the sorted-order check, which does not apply to unsorted output" % order_na)
        low_level.append(", ".join(oracle_parts) + ".")
        if s3_pass or s3_failed or order_na:
            s3_parts = []
            if s3_pass:
                s3_parts.append("%d cells passed the order check" % s3_pass)
            if s3_none:
                s3_parts.append("%d of them carry control_name none because the control's unsorted output fails the same check and the harness recorded no control reading" % s3_none)
            if s3_failed:
                s3_parts.append("%d cells failed it (see the negative findings)" % s3_failed)
            if order_na:
                s3_parts.append("%d cells failed only the order check, not applicable to unsorted output" % order_na)
            low_level.append("Phase 3b sorted-output check (%s): %s." % (", ".join(sorted(s3_nodes)) or "B200", ", ".join(s3_parts)))
        low_level.extend(cav.get("low_level", []))
        negative = list(cav["negative"])
        if not negative:
            negative.append("No cell where this adapter was the winner changed hands against the control in the same lock section. Deficits above are against other adapters of the same scope.")
        surprises = list(cav["surprises"]) or ["None beyond what the tables show."]
        open_q = list(cav["open_questions"]) + [GENERIC_OPEN_B200 if args.arch == "b200" else GENERIC_OPEN_GB200]

        findings = {
            "author": "eval-engineer", "approach_id": approach, "variant": variant,
            "contract": {
                "input_location": "registers", "input_arrangement": ", ".join(sorted(a for a in arr if a)) or "see header",
                "output_location": "smem", "output_form": ", ".join(sorted(o for o in outf if o)) or "indices",
                "output_sorted": bool(sorted_out),
                "temp_storage_bytes": int(max(ts_bytes)) if ts_bytes else 0,
                "sync_assumed": "harness group barrier between calls that reuse the temp storage and the output buffers",
                "sync_performed": "internal synchronization as declared in the adapter header, measured inside the call",
                "header_contract": contract_text or "see the adapter header",
                "k_range": "measured k %d to %d" % (min(ks), max(ks)) if ks else "see header",
                "n_range": "measured N %d to %d, scope %s" % (min(ns), max(ns), scope) if ns else "see header",
                "type_constraints": "measured on " + ", ".join(sorted(keys)) + " keys with int32 indices",
                "nan_behavior": "focus-set inputs contain no NaN, see the adapter header",
                "tie_break_support": "prefer_smaller_index measured" if any(k[2] == "prefer_smaller_index" for k in standing) else "not_guaranteed only",
                "sorted_output_support": cav.get("sorted_support") or ("inherent, passed the phase 3b order check" if sorted_out else "not sorted (see findings)"),
            },
            "findings": {"low_level": low_level, "algorithmic": algorithmic, "negative": negative, "surprises": surprises, "open_questions": open_q},
            "techniques_adopted": [], "techniques_contributed": [],
            "report_path": report,
            "source_paths": ["harness/adapters/%s.cuh" % aid],
            "correctness_notes": "phase 3 final evaluation, harness %s, nodes %s" % (", ".join(sorted(hv)), ", ".join(sorted(n for n in nodes if n)))
                                 + ((", " + cav["correctness"]) if cav.get("correctness") else ""),
            "unsorted_by_contract": bool(cav.get("unsorted_by_contract")),
        }
        fpath = os.path.join(fdir, "%s_%s.json" % (aid, args.arch))
        json.dump(findings, open(fpath, "w"), indent=1)
        out = os.path.join(SWARM, "cards", "%s_%s_%s_%s_1.json" % (approach, variant, args.arch, args.date))
        cmd = [sys.executable, os.path.join(HERE, "make_card.py"), "--findings", fpath, "--source-commit", adapter_commit(aid), "--out", out]
        for f in files:
            cmd += ["--batch", f]
        if args.dry_run:
            print("would write", out, "from", len(files), "batches")
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        print((r.stdout.strip() or r.stderr.strip())[-200:])
        if r.returncode == 0:
            written.append(out)
    print("%d cards written" % len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
