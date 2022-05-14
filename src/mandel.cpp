#include <cstddef>
#include <memory>

#include <png.h>

//#include <CLI/CLI.hpp>
//#include <spdlog/spdlog.h>
#include "GPU/harris_gpu.hh"
#include "CPU/harris_cpu.hh"
#include "png/png_handler.hh"

// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  char filename[] = "../img/b001.png";
  char mode[] = "GPU";
/*

  CLI::App app{"harris"};

  app.add_option("-o", filename, "Input image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

  CLI11_PARSE(app, argc, argv);*/

  PNG_data image_data = read_png_file(filename);

  if (mode == "CPU") {
	detect_point(image_data);
  }
  else if (mode == "GPU")
  {
    // render(reinterpret_cast<char*>(buffer.get()), width, height, stride, niter);
	gpu::detect_point(image_data);
  }

  // Save
  //write_png(buffer.get(), width, height, stride, filename.c_str());
  //spdlog::info("Output saved in {}.", filename);
}

