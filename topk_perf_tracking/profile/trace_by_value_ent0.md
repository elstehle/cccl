# Kernel trace -- `by_value (entropy=0.000)`

- kernels: **7**
- sum of kernel durations: **11208.01 us**
- wall-time span (first start -> last end): **11211.62 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 20.19 | `device_segmented_topk_histogram_kernel` |
| 2 | 20.90 | 2.62 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 24.35 | 18.18 | `device_segmented_topk_filter_kernel` |
| 4 | 43.20 | 2.75 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 46.66 | 16.86 | `device_segmented_topk_filter_kernel` |
| 6 | 63.81 | 2.62 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 66.85 | 11144.77 | `device_segmented_topk_last_filter_kernel` |
