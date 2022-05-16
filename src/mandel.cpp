#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "GPU/harris_gpu.hh"
#include "CPU/harris_cpu.hh"
#include "png/png_handler.hh"

#include "GPU-CudaStream/harris_gpu.hh"
#include "GPU-DilateSharedMEM/harris_gpu.hh"
#include "GPU-kernelSharedMem/harris_gpu.hh"

// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  std::string filename = "../img/b001.png";
  std::string mode = "CPU";

  CLI::App app{"harris"};

  app.add_option("-o", filename, "Input image");
  app.add_set("-m", mode, {"GPU", "CPU", "GPUStream", "GPUDilate", "GPUKernel"}, "Either 'GPU' or 'CPU' or 'GPUStream' or 'GPUDilate' or 'GPUKernel'");

  CLI11_PARSE(app, argc, argv);


  spdlog::info("Running {} mode on file {}", mode, filename);
  if (mode == "CPU") {
	PNG_data image_data = read_png_file(filename.c_str());
	detect_point(image_data);
  }
  else if (mode == "GPU")
  {
	PNG_data image_data = read_png_file(filename.c_str());
	gpu::detect_point(image_data);
  }
  else if (mode == "GPUStream")
  {
	PNG_data image_data = read_png_file(filename.c_str());
	gpuCudaStream::detect_point(image_data);
  }
  else if (mode == "GPUDilate")
  {
	PNG_data image_data = read_png_file(filename.c_str());
	gpuDilateSharedMem::detect_point(image_data);
  }
  else if (mode == "GPUKernel")
  {
	PNG_data image_data = read_png_file(filename.c_str());
	gpuKernelSharedMem::detect_point(image_data);
  }

  // Save
  //write_png(buffer.get(), width, height, stride, filename.c_str());
  //spdlog::info("Output saved in {}.", filename);
}

