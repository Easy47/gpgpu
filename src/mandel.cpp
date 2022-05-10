#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "harris_cpu.hpp"


// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  std::string filename = "";
  std::string mode = "GPU";


  CLI::App app{"harris"};

  app.add_option("-o", filename, "Input image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

  CLI11_PARSE(app, argc, argv);


  if (mode == "CPU") {
    detect_point(filename);
  }
  /*else if (mode == "GPU")
  {
    render(reinterpret_cast<char*>(buffer.get()), width, height, stride, niter);
  }*/

  // Save
  //write_png(buffer.get(), width, height, stride, filename.c_str());
  //spdlog::info("Output saved in {}.", filename);
}

