#include "CPU/harris_cpu.hh"
#include <vector>
#include <benchmark/benchmark.h>
#include <iostream>


void BM_Rendering_cpu(benchmark::State& st)
{
   std::cout << "Benchmark starting..." << std::endl;
  //int stride = width * kRGBASize;
  //std::vector<char> data(height * stride);
  char filename[] = "../img/b001.png";
  PNG_data image_data = read_png_file(filename);
  for (auto _ : st)
    detect_point(image_data);

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
