# Kernel trace -- `lift_part (entropy=1.000)`

- kernels: **7**
- sum of kernel durations: **74.82 us**
- wall-time span (first start -> last end): **78.34 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 22.53 | `device_segmented_topk_histogram_kernel` |
| 2 | 23.23 | 2.75 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 26.82 | 21.66 | `device_segmented_topk_filter_kernel` |
| 4 | 49.18 | 3.07 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 52.96 | 10.24 | `device_segmented_topk_filter_kernel` |
| 6 | 63.49 | 7.71 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 71.49 | 6.85 | `device_segmented_topk_last_filter_kernel` |
