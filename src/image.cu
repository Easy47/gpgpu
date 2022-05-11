#include "image.hh"
#include <cstdlib>
#include <iostream>
#include <math.h>

__host__ gray8_image::gray8_image(int height, int width, png_bytep *row_pointers) {
    sx = height;
    sy = width;
    length = sx * sy;

    auto rc = cudaMalloc(&pixels, sizeof(double) * length);//new double[length];
    if (rc)
        abortError("Fail buffer allocation in gray8_image");

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            png_bytep pixel = &(row_pointers[i][j * 4]);
            auto r = pixel[0];
            auto g = pixel[1];
            auto b = pixel[2];
            auto transp = pixel[3];
            pixels[i * width + j] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
}

__host__ gray8_image::gray8_image(int _sx, int _sy) {
    this->sx = _sx;
    this->sy = _sy;

    this->length = sx * sy;
    //this->pixels = new double[length];
    auto rc = cudaMalloc(&pixels, sizeof(double) * length);//new double[length];
    if (rc)
        abortError("Fail buffer allocation in gray8_image(empty)");
}

__host__ gray8_image::~gray8_image() {
    auto rc = cudaFree(pixels);
    if (rc)
        abortError("Fail buffer free in gray8_image");
    //delete [] pixels;
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

__device__ gray8_image *img_add(gray8_image *img, gray8_image *img2, gray8_image *res_img) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(img->width/dimBlock.x, img->height/dimBlock.y);
    kvecAdd<<<dimBlock,dimGrid>>>(img->pixels, img2->pixels, res_img->pixels, img->length);
    return res_img;
}

__global__ void kvecMultScalar(double *img1, int val, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] * val;
}

__device__ gray8_image *img_mult_scalar(gray8_image *img, int val, gray8_image *res_img) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(img->width/dimBlock.x, img->height/dimBlock.y);
    kvecMultScalar<<<dimBlock,dimGrid>>>(img->pixels, val, res_img->pixels, img->length);
    return res_img;
}

__global__ void kvecMult(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] * img2[i];
}


gray8_image *img_mult(gray8_image *img, gray8_image *img2, gray8_image *res_img) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(img->width/dimBlock.x, img->height/dimBlock.y);
    kvecMult<<<dimBlock,dimGrid>>>(img->pixels, img2->pixels, res_img->pixels, img->length);
    return res_img;
}

__global__ void kvecDiv(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] / img2[i];
}

gray8_image *img_div(gray8_image *img, gray8_image *img2, gray8_image *res_img) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(img->width/dimBlock.x, img->height/dimBlock.y);
    kvecDiv<<<dimBlock,dimGrid>>>(img->pixels, img2->pixels, res_img->pixels, img->length);
    return res_img;
}

__global__ void kvecSous(double *img1, double *img2, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] - img2[i];
}

gray8_image *img_sous(gray8_image *img, gray8_image *img2, gray8_image *res_img) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(img->width/dimBlock.x, img->height/dimBlock.y);
    kvecSous<<<dimBlock,dimGrid>>>(img->pixels, img2->pixels, res_img->pixels, img->length);
    return res_img;
}

__global__ void kvecAddScalar(double *img1, int value, double *res_img, int lgt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lgt)
	return;
    res_img[i] = img1[i] + val;
}

gray8_image *img_add_scalar(gray8_image *img, int value, gray8_image *res_img) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(img->width/dimBlock.x, img->height/dimBlock.y);
    kvecAddScalar<<<dimBlock,dimGrid>>>(img->pixels, value, res_img->pixels, img->length);
    return res_img;
}


gray8_image *gray8_image::gray_convolution(gray8_image* masque, gray8_image *res_img) {
    int index = (masque->sx - 1) / 2;
    //gray8_image *res_img = new gray8_image(this->sx, this->sy);
    for (int x = 0; x < this->sx; x++) {
        for (int y = 0; y < this->sy; y++) {
            double res = 0;
            for (int i = -index; i <= index; i++) {
                if (i + x < 0 || i + x >= this->sx) {
                    continue;
                }
                for (int j = -index; j <= index; j++) {
                    if (j + y < 0 || j + y >= this->sy) {
                        continue;
                    }

                    double m = masque->pixels[(i + index) * masque->sy + (j + index)];
                    double n = this->pixels[(x + i) * this->sy + (y + j)];
                    res += m * n;
                }
            }
            res_img->pixels[x * this->sy + y] = res;
        }
    }
    return res_img;
}

gray8_image *gray8_image::gray_convolution(gray8_image* masque) {
    int index = (masque->sx - 1) / 2;
    gray8_image *res_img = new gray8_image(this->sx, this->sy);
    for (int x = 0; x < this->sx; x++) {
        for (int y = 0; y < this->sy; y++) {
            double res = 0;
            for (int i = -index; i <= index; i++) {
                if (i + x < 0 || i + x >= this->sx) {
                    continue;
                }
                for (int j = -index; j <= index; j++) {
                    if (j + y < 0 || j + y >= this->sy) {
                        continue;
                    }

                    double m = masque->pixels[(i + index) * masque->sy + (j + index)];
                    double n = this->pixels[(x + i) * this->sy + (y + j)];
                    res += m * n;
                }
            }
            res_img->pixels[x * this->sy + y] = res;
        }
    }
    return res_img;
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
