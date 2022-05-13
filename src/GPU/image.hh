#pragma once
//TIFO
#include <png.h>

class gray8_image {

  public:
            /**
             * Image creation and allocation.
             * @param sx width of the image in pixel
             * @param sy height of the image in pixel
            */
            gray8_image(int height, int width, png_bytep *row_pointers);
            gray8_image(int sx, int sy);
            ~gray8_image();

            /**
             * Gives the pixel buffer aligned according to TL_IMAGE_ALIGNMENT
             * macro.
             * @return the pixel buffer.
             */
             const double*& get_buffer() const;

            /**
             * Gives the pixel buffer aligned according to TL_IMAGE_ALIGNMENT
             * macro.
             * @return the pixel buffer.
             */
            double*& get_buffer();
	    void get_data_from(double *input);

            //void gray_convolution(gray8_image* masque, gray8_image *res_img);
            gray8_image *dilate(gray8_image* masque);
            float min();
            float max();

  public:
            /**Width of the image in pixels.*/
            int sx;
            /**Height of the image in pixels.*/
            int sy;
            /**Size of the reserved area in bytes.*/
            int length;
            /**Buffer*/
            double* pixels;
};
__global__ void kvecAdd(double *img1, double *img2, double *res_img, int lgt);
__global__ void kvecMultScalar(double *img1, int val, double *res_img, int lgt);
__global__ void kvecMult(double *img1, double *img2, double *res_img, int lgt);
__global__ void kvecDiv(double *img1, double *img2, double *res_img, int lgt);
__global__ void kvecSous(double *img1, double *img2, double *res_img, int lgt);
__global__ void kvecAddScalar(double *img1, int value, double *res_img, int lgt);
__global__ void kvecConvol(double *img, int img_x, int img_y, double *mask, int msk_size, double *res_img);
__global__ void kvecDilate(double *img, int img_x, int img_y, double *mask, int msk_size, double *res_img);
