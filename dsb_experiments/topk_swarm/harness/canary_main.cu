// Standalone drift canary: run first and last in every measurement batch (run.py does this).
// Prints one JSON object with the fixed shuffle-chain cycles per op and the fixed FMA kernel time.
#include "topk_harness.cuh"

int main(int argc, char** argv)
{
  std::string out;
  for (int i = 1; i + 1 < argc; ++i)
  {
    if (std::string(argv[i]) == "--out")
    {
      out = argv[i + 1];
    }
  }
  topk::device_info dev = topk::query_device();
  topk::canary_values c = topk::run_canary(dev);
  topk::json j          = topk::json::object();
  j.add("timestamp", topk::now_iso());
  j.add("device", dev.name);
  j.add("sms", dev.sms);
  j.add("cuda_runtime", topk::cuda_version_string(dev.runtime));
  j.add("cuda_driver", topk::cuda_version_string(dev.driver));
  j.add("lat_cycles_per_op", c.lat_cycles_per_op);
  j.add("thr_us", c.thr_us);
  const std::string s = j.str();
  printf("%s\n", s.c_str());
  if (!out.empty())
  {
    FILE* f = fopen(out.c_str(), "w");
    if (f)
    {
      fwrite(s.data(), 1, s.size(), f);
      fputc('\n', f);
      fclose(f);
    }
  }
  return 0;
}
