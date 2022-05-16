#include "image.hh"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <spdlog/spdlog.h>
#include <stdio.h>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

namespace ImageGpuKernelSharedMem {

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  std::cout << "error : " << line << msg << std::endl;
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


__host__ gray8_image::gray8_image(int height, int width, png_bytep *row_pointers) {
    sx = height;
    sy = width;
    length = sx * sy;

    auto rc = cudaMalloc(&pixels, sizeof(double) * length);//new double[length];

    if (rc)
        abortError("Fail buffer allocation in gray8_image");

    double *buffer = new double[length];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            png_bytep pixel = &(row_pointers[i][j * 4]);
            auto r = pixel[0];
            auto g = pixel[1];
            auto b = pixel[2];
            auto transp = pixel[3];
            buffer[i * width + j] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
    get_data_from(buffer);
	delete buffer;
}

__host__ gray8_image::gray8_image(int _sx, int _sy) {
    this->sx = _sx;
    this->sy = _sy;

    this->length = sx * sy;
    auto rc = cudaMallocManaged(&pixels, sizeof(double) * length);
    if (rc)
        abortError("Fail buffer allocation in gray8_image");
}

__host__ gray8_image::~gray8_image() {
    auto rc = cudaFree(pixels);
    if (rc)
        abortError("Fail buffer free in gray8_image");
}

__host__ void gray8_image::get_data_from(double *input) {
    cudaMemcpy(pixels, input, length * sizeof(double), cudaMemcpyHostToDevice);
}

__device__ double *&gray8_image::get_buffer() {
    return pixels;
}

__global__ void kvecAdd(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] + img2[i];
}


__global__ void kvecMultScalar(double *img1, int val, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] * val;
}


__global__ void kvecMult(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] * img2[i];
}



__global__ void kvecDiv(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] / img2[i];
}


__global__ void kvecSous(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] - img2[i];
}


__global__ void kvecAddScalar(double *img1, int value, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] + value;
}


__global__ void kvecConvol(double *img, int img_x, int img_y, double *mask, int msk_size, double *res_img) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= img_x)
	return;
    if (y >= img_y)
	return;

    int index = (msk_size - 1) / 2;
    double res = 0;
    for (int i = -index; i <= index; i++) {
        if (i + x < 0 || i + x >= img_x) {
            continue;
        }
        for (int j = -index; j <= index; j++) {
            if (j + y < 0 || j + y >= img_y) {
                continue;
            }

            double m = mask[(i + index) * msk_size + (j + index)];
            double n = img[(x + i) * img_y + (y + j)];
            res += m * n;
        }
    }
    res_img[x * img_y + y] = res;
}


__global__ void kvecDilate(double *img, int img_x, int img_y, double *mask, int msk_size, double *res_img) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= img_x)
        return;
    if (y >= img_y)
        return;

    int tIdX = threadIdx.x;
    int tIdY = threadIdx.y;

    int tIdX = threadIdx.x;
    int tIdY = threadIdx.y;

    __shared__ double maskShared[625];
    if (tIdX * 25 + tIdY < 625) {
        maskShared[tIdX * 25 + tIdY] = mask[tIdX * 25 + tIdY];
    }

    __shared__ double imgShared[1024];
    imgShared[tIdX * 32 + tIdY] = img[x * img_y + y];
    __syncthreads();

    int tileStartX = blockDim.x * blockIdx.x;
    int tileEndX = (blockDim.x + 1) * blockIdx.x;
    int tileStartY = blockDim.y * blockIdx.y;
    int tileEndY = (blockDim.y + 1) * blockIdx.y;
    int index = (msk_size - 1) / 2;
    double max = img[x * img_y + y];
    for (int i = -index; i <= index; i++) {
        if (i + x < 0 || i + x >= img_x) {
            continue;
        }
        for (int j = -index; j <= index; j++) {
            if (j + y < 0 || j + y >= img_y) {
                continue;
            }

            double m = mask[(i + index) * msk_size + (j + index)];
            if (m == 0) {
                continue
            }
            double n;
            if ((x + i) > tileStartX && (x + i) < tileEndX && (y + j) > tileStartY && (y + j) < tileEndY) {
                n = imgShared[(tIdX + i) * 32 + (tIdY + j)];
            }
            else {
                n = img[(x + i) * img_y + (y + j)];
            }
            if (n > max)
                max = n;
        }
    }
    res_img[x * img_y + y] = max;
}

gray8_image *gray8_image::dilate(gray8_image* masque) {
    int index = (masque->sx - 1) / 2;
    gray8_image *res_img = new gray8_image(this->sx, this->sy);
    for (int x = 0; x < this->sx; x++) {
        for (int y = 0; y < this->sy; y++) {
            double max = this->pixels[x * this->sy + y];
            for (int i = -index; i <= index; i++) {
                if (i + x < 0 || i + x >= this->sx) {
                    continue;
                }
                for (int j = -index; j <= index; j++) {
                    if (j + y < 0 || j + y >= this->sy) {
                        continue;
                    }

                    double m = masque->pixels[(i + index) * masque->sy + (j + index)];
                    if (m == 0) {
                        continue;
                    }
                    double n = this->pixels[(x + i) * this->sy + (y + j)];
                    if (n > max) {
                        max = n;
                    }
                }
            }
            res_img->pixels[x * this->sy + y] = max;
        }
    }
    return res_img;
}

float gray8_image::max() {
    float res = this->pixels[0];
    for (int i = 0; i < this->sx; i++) {
        for (int j = 0; j < this->sy; j++) {
            if (this->pixels[i * this->sy + j] > res) {
                res = this->pixels[i * this->sy + j];
            }
        }
    }
    return res;
}

float gray8_image::min() {
    float res = this->pixels[0];
    for (int i = 0; i < this->sx; i++) {
        for (int j = 0; j < this->sy; j++) {
            if (this->pixels[i * this->sy + j] < res) {
                res = this->pixels[i * this->sy + j];
            }
        }
    }
    return res;
}
}
