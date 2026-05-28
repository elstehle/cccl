# Kernel trace -- `main / I32-I32 / 2^28 / sel=2^8 / entropy=0.201`

- kernels: **4**
- sum of kernel durations: **436.23 us**
- wall-time span (first start -> last end): **444.51 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 232.67 | `DeviceTopKHistogramKernel` |
| 2 | 232.93 | 199.90 | `DeviceTopKKernel` |
| 3 | 433.09 | 1.82 | `DeviceTopKKernel` |
| 4 | 442.69 | 1.82 | `DeviceTopKLastFilterKernel` |
