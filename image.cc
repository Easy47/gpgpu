#include "image.hh"
#include <cstdlib>
#include <iostream>
#include <math.h>

gray8_image::gray8_image(int height, int width, png_bytep *row_pointers) {
    sx = height;
    sy = width;
    length = sx * sy;
    pixels = new double[length];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            png_bytep pixel = &(row_pointers[i][j * 4]);
            auto r = pixel[0];
            auto g = pixel[1];
            auto b = pixel[2];
            auto transp = pixel[3];
            pixels[i * width + j] = 0.299 * r + 0.587 * g + 0.114 * b;;
        }
    }
}

gray8_image::gray8_image(int _sx, int _sy) {
    this->sx = _sx;
    this->sy = _sy;

    this->length = sx * sy;
    this->pixels = new double[length];
}

gray8_image::~gray8_image() {
    delete [] pixels;
}

double *&gray8_image::get_buffer() {
    return pixels;
}

gray8_image *img_mult_scalar(gray8_image *img, int val) {
    gray8_image *res = new gray8_image(img->sx, img->sy);
    for (int i = 0; i < img->sx; i++) {
        for (int j = 0; j < img->sy; j++) {
            res->pixels[i * img->sy + j] = img->pixels[i * img->sy + j] * val;
        }
    }
    return res;
}


gray8_image *img_mult(gray8_image *img, gray8_image *img2) {
    gray8_image *res = new gray8_image(img->sx, img->sy);
    for (int i = 0; i < img->sx; i++) {
        for (int j = 0; j < img->sy; j++) {
            res->pixels[i * img->sy + j] = img2->pixels[i * img->sy + j] * img->pixels[i * img->sy + j];
        }
    }
    return res;
}

gray8_image *img_div(gray8_image *img, gray8_image *img2) {
    gray8_image *res = new gray8_image(img->sx, img->sy);
    for (int i = 0; i < img->sx; i++) {
        for (int j = 0; j < img->sy; j++) {
            res->pixels[i * img->sy + j] = img->pixels[i * img->sy + j] / img2->pixels[i * img->sy + j];
        }
    }
    return res;
}

gray8_image *img_add(gray8_image *img, gray8_image *img2) {
    gray8_image *res = new gray8_image(img->sx, img->sy);
    for (int i = 0; i < img->sx; i++) {
        for (int j = 0; j < img->sy; j++) {
            res->pixels[i * img->sy + j] = img2->pixels[i * img->sy + j] + img->pixels[i * img->sy + j];
        }
    }
    return res;
}

gray8_image *img_sous(gray8_image *img, gray8_image *img2) {
    gray8_image *res = new gray8_image(img->sx, img->sy);
    for (int i = 0; i < img->sx; i++) {
        for (int j = 0; j < img->sy; j++) {
            res->pixels[i * img->sy + j] = img->pixels[i * img->sy + j] - img2->pixels[i * img->sy + j];
        }
    }
    return res;
}

gray8_image *img_add_scalar(gray8_image *img, int value) {
    gray8_image *res = new gray8_image(img->sx, img->sy);
    for (int i = 0; i < img->sx; i++) {
        for (int j = 0; j < img->sy; j++) {
            res->pixels[i * img->sy + j] = img->pixels[i * img->sy + j] + value;
        }
    }
    return res;
}


gray8_image *gray8_image::gray_convolution(int *masque)
{

    gray8_image *res_img = new gray8_image(this->sx, this->sy);
    for (int y = 1; y < this->sy - 1; y++)
    {
        for (int x = 1; x < this->sx - 1; x++)
        {
            int res = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int m = masque[(i + 1) * 3 + (j + 1)];
                    int n = this->pixels[(y + i) * this->sx + (x + j)];
                    res += m * n;
                }
            }
            res = std::min(255, res);
            res = std::max(0, res);
            res_img->get_buffer()[y * this->sx + x] = res;
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
