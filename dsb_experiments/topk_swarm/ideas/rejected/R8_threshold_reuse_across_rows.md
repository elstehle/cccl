# R8: Threshold reuse across rows in the batched setting
status: rejected across rows, accepted within a row (2026-10-25, algo-researcher)
scope: block, device
seed: the lead's list (threshold reuse across rows for the batched setting)
idea: use the k-th value of the previous row (previous token, previous decode step) as the filter threshold for the next row.
reason: across independent rows the previous threshold is a guess, not a bound, so it needs count verification, a retry ladder and an overflow fallback, and a stale guess is expensive: the temporal-hint GVR V1 lost 3.6 to 6x on an invalid hint, which is why V2 went hint-free (`reference/GVR_V2_ALGORITHM_REPORT.md` section 1). Router logits of consecutive tokens are not correlated enough to make the guess reliable, and the harness's rows are independent random draws, so nothing could be measured here anyway. Within one row the threshold of the prefix is an exact lower bound for the rest, which is PR #9281's buffered algorithm at warp scope and N3 at block scope.
what survives: N3 (running exact threshold over chunks of the same row), W5 explore item 5 (note for the device level).
