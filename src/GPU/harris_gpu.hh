#pragma once
#include <cstddef>
#include <memory>
#include "png_handler.hh"

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
//extern "C"
void detect_point(PNG_data image_data);

//void harris_gpu(char* buffer, int width, int height, std::ptrdiff_t stride, int n_iterations = 100);
