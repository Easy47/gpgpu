#include "harris_cpu.hpp"
#include <vector>
#include <benchmark/benchmark.h>


void BM_Rendering_cpu(benchmark::State& st)
{
  //int stride = width * kRGBASize;
  //std::vector<char> data(height * stride);
  char *filename = ""
  for (auto _ : st)
    detect_harris_points(filename);

  //st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

/*void BM_Rendering_gpu(benchmark::State& st)
{
  int stride = width * kRGBASize;
  std::vector<char> data(height * stride);

  for (auto _ : st)
    render(data.data(), width, height, stride, niteration);

  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}
*/
BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

/*BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();*/

BENCHMARK_MAIN();
