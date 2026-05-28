# Kernel trace -- `byval`

- kernels: **7**
- sum of kernel durations: **74.91 us**
- wall-time span (first start -> last end): **78.43 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 22.34 | `device_segmented_topk_histogram_kernel` |
| 2 | 23.04 | 2.91 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 26.78 | 21.31 | `device_segmented_topk_filter_kernel` |
| 4 | 48.80 | 3.10 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 52.61 | 10.05 | `device_segmented_topk_filter_kernel` |
| 6 | 62.94 | 7.71 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 70.94 | 7.49 | `device_segmented_topk_last_filter_kernel` |
