# Kernel trace -- `lift_part / I32-I32 / 2^28 / sel=2^8 / entropy=0.201`

- kernels: **7**
- sum of kernel durations: **1076.87 us**
- wall-time span (first start -> last end): **1079.01 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 235.65 | `device_segmented_topk_histogram_kernel` |
| 2 | 235.94 | 4.22 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 240.42 | 237.86 | `device_segmented_topk_filter_kernel` |
| 4 | 478.82 | 4.22 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 483.58 | 232.67 | `device_segmented_topk_filter_kernel` |
| 6 | 716.51 | 4.03 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 720.80 | 358.21 | `device_segmented_topk_last_filter_kernel` |
