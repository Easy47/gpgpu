
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

            gray8_image *gray_convolution(int* masque);
            gray8_image *gray_convolution(gray8_image* masque);
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
    gray8_image *img_mult(gray8_image *img, gray8_image *img2);
    gray8_image *img_add(gray8_image *img, gray8_image *img2);
    gray8_image *img_sous(gray8_image *img, gray8_image *img2);
    gray8_image *img_mult_scalar(gray8_image *img, int val);
    gray8_image *img_div(gray8_image *img, gray8_image *img2);
    gray8_image *img_add_scalar(gray8_image *img, int value);




