#include <iostream> 
#include <cmath>
#include "image.hh"
#include <png.h>
#include <cstddef>
#include <algorithm>
#include <vector>
int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

/*void write_png(float* buffer,
               int width,
               int height,
               int stride,
               const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_GRAY,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr,  reinterpret_cast<png_const_bytep>(reinterpret_cast<std::byte*>(buffer)));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}*/

void read_png_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        abort();

    png_infop info = png_create_info_struct(png);
    if (!info)
        abort();

    if (setjmp(png_jmpbuf(png)))
        abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    if (row_pointers)
        abort();

    row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
    {
        row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png, info));
    }
    png_read_image(png, row_pointers);
    png_destroy_read_struct(&png, &info, NULL) ;
}

void mgrid(int v11, int v12, int v21, int v22, gray8_image *t1, gray8_image *t2) {

    int value1 = v11;
    for (int i = 0; i < t1->sx; i++) {
        for (int j = 0; j < t1->sy; j++) {
            t1->pixels[i * t1->sx + j] = value1;
        }
        value1++;
    }
    for (int i = 0; i < t2->sx; i++) {
        int value2 = v21;
        for (int j = 0; j < t2->sy; j++) {
            t2->pixels[i * t2->sx + j] = value2;
            value2++;
        }
    }
}

void gauss_derivative_kernels(int size, int sizey, gray8_image *gx, gray8_image *gy) {

    sizey = size;
    int size1 = std::abs(-size) + std::abs(size) + 1;
    int size2 = std::abs(-sizey) + std::abs(sizey) + 1;


    //gray8_image *gx = new gray8_image(size1, size1);
    //gray8_image *gy = new gray8_image(size1, size2);

    //float **gx = new float *[size1];
    //float **gy = new float *[size1];

    gray8_image *y = new gray8_image(size1, size1);
    gray8_image *x = new gray8_image(size1, size1);

    mgrid(-size, size + 1, -sizey, sizey + 1, y, x);

    double val1 = (0.33*size) * (0.33*size) * 2;
    double val2 = (0.33*sizey) * (0.33*sizey) * 2;

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size1; j++) {
            gx->pixels[i * size1 + j] = -1 * x->pixels[i * size1 + j] * std::exp(-1 * ((x->pixels[i * size1 + j] * x->pixels[i * size1 + j] / val1) + (y->pixels[i * size1 +j] * y->pixels[i * size1 + j] / val1)));
            gy->pixels[i * size1 + j]= -1 * y->pixels[i * size2 + j] * std::exp(-1 * ((x->pixels[i * size1 + j] * x->pixels[i * size1 + j] / val1) + (y->pixels[i * size1 + j] * y->pixels[i * size1 + j] / val1)));
        }
    }
}
void gauss_kernel(int size, int sizey, gray8_image *res) {

    int size1 = std::abs(-size) + std::abs(size) + 1;
    int size2 = std::abs(-sizey) + std::abs(sizey) + 1;

    //float **y = new float *[size1];
    //float **x = new float *[size2];

    gray8_image *y = new gray8_image(size1, size1);
    gray8_image *x = new gray8_image(size1, size2);

    mgrid(-size, size + 1, -sizey, sizey + 1, y, x);
    
    double val1 = (0.33*size) * (0.33*size) * 2;
    double val2 = (0.33*sizey) * (0.33*sizey) * 2;

    for (int i = 0; i < size2; i++) {
        for (int j = 0; j < size2; j++) {
            y->pixels[i * size1 + j] *= y->pixels[i * size1 + j] / val2;
        }
    }

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size1; j++) {
            x->pixels[i * size1 + j] *= x->pixels[i * size1 + j] / val1; 
        }
    }

    //float **res = new float *[size1];
    //gray8_image *res = new gray8_image(size1, size1);

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size1; j++) {
            res->pixels[i * size1 + j] = std::exp(-(x->pixels[i * size1 + j] + y->pixels[i * size1 + j]));
            //std::cout << res->pixels[i * size1 + j] << " ";
        }
        //std::cout << "\n";
    }
}

void gauss_derivatives(gray8_image *img, int size, gray8_image *imx, gray8_image *imy) {

    gray8_image *gx = new gray8_image(2*size + 1, 2*size + 1);
    gray8_image *gy = new gray8_image(2*size + 1, 2*size + 1);
    gauss_derivative_kernels(size, size, gx, gy);
    std::cout << gx->pixels[5 * gx->sy + 5] << "\n";
    std::cout << img->pixels[100 * img->sy + 250] << "\n";
    imx->pixels = img->gray_convolution(gx)->pixels;
    std::cout << imx->pixels[200 * imx->sy + 20] << "\n";
    imy->pixels = img->gray_convolution(gy)->pixels;
}

gray8_image *compute_harris_response(gray8_image *img) {
    int DERIVATIVE_KERNEL_SIZE = 3;
    int OPENING_SIZE = 3;

    gray8_image *imx = new gray8_image(img->sx, img->sy);
    gray8_image *imy = new gray8_image(img->sx, img->sy);
    gauss_derivatives(img, DERIVATIVE_KERNEL_SIZE, imx, imy);
    gray8_image *gauss = new gray8_image(2*OPENING_SIZE + 1, 2*OPENING_SIZE + 1);
    gauss_kernel(OPENING_SIZE, OPENING_SIZE, gauss);

    gray8_image *imx2 = img_mult(imx, imx);

    auto Wxx = imx2->gray_convolution(gauss);
    gray8_image *imximy = img_mult(imx, imy);   
    auto Wxy = imximy->gray_convolution(gauss);
    gray8_image *imy2 = img_mult(imy, imy);

    auto Wyy = imy2->gray_convolution(gauss);

    auto Wdet = img_sous(img_mult(Wxx, Wyy),img_mult(Wxy, Wxy));

    auto Wtr = img_add(Wxx, Wyy);
    gray8_image *tmp = img_add_scalar(Wtr, 1);
    gray8_image *res = img_div(Wdet, tmp);
    return res;

}
class Point {
    public:
    int x;
    int y;
    float val;
    Point(int x, int y, float val) {
        this->x = x;
        this->y = y;
        this->val = val;
    }
};

std::vector<Point> compute_mask(gray8_image *harris_resp, gray8_image *t2, float threshold) {
    std::vector<Point> res;
    float rtol = 0.0001;
    float atol = 0.0000001;
    float max = t2->max();
    float min = t2->min();
    for (int i = 0; i < harris_resp->sx; i++) {
        for (int j = 0; j < harris_resp->sy; j++) {
            if ((abs(harris_resp->pixels[i * harris_resp->sy + j] - t2->pixels[i * harris_resp->sy + j]) <= atol * rtol * abs(t2->pixels[i * harris_resp->sy + j])) 
            && (harris_resp->pixels[i * harris_resp->sy + j] > min + threshold * (max - min))) {
                //mask->pixels[i * harris_resp->sy + j] = 1;
                Point tmp = Point(i , j, harris_resp->pixels[i * harris_resp->sy + j]);
                res.push_back(tmp);
            }/* else {
                mask->pixels[i * harris_resp->sy + j] = 0;
            }*/
        }
    }
    return res;
}

bool myfunction (Point p1,Point p2) { return ( p1.val < p2.val); }


void detect_harris_points(gray8_image *image_gray, int max_keypoints = 30, int min_distance = 25, float threshold = 0.1) {
    gray8_image *harris_resp = compute_harris_response(image_gray);
    float tmp[625] = { 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                };
    std::cout << "HARRIS RESP : " << "\n";
    std::cout << harris_resp->pixels[200 * harris_resp->sy + 300] << "\n";
    gray8_image *ellipse_kernel = new gray8_image(25,25);
    for (int i = 0; i < 625; i++) {
        ellipse_kernel->pixels[i] = tmp[i];
    }
    std::cout << harris_resp->max() << "\n";
    gray8_image *dilate = harris_resp->dilate(ellipse_kernel);
    std::cout << "dilate : " << "\n";
    std::cout << dilate->pixels[89 * dilate->sy + 250] << "\n";
    std::cout << dilate->max() << "\n";
    //gray8_image *mask = new gray8_image(image_gray->sx, image_gray->sy);
    std::vector<Point> candidate = compute_mask(dilate, harris_resp, threshold);
    std::sort(candidate.begin(), candidate.end(), myfunction);
    std::cout << candidate[0].x  << " " << candidate[0].y << " " << candidate[0].val << "\n";
    std::cout <<  candidate[1].x  << " " << candidate[1].y  << " " << candidate[1].val <<  "\n";
    std::cout <<  candidate[35].x  << " " << candidate[35].y  << " " << candidate[35].val <<  "\n";
    std::cout << candidate.size() << "\n";
    std::vector<Point> res;
    int nb = 0;
    for (auto i = candidate.begin(); i != candidate.end(); i++) {
        if (nb == max_keypoints) {
            break;
        }
        nb++;
        res.push_back(*i);
    }

    for (auto i = res.begin(); i != res.end(); i++) {
        std::cout << (*i).x << " " << (*i).y << " " << (*i).val <<  "\n";
    }


    
}
int main(int argc, char *argv[]) {
    //gauss_kernel(5, 5);
    /*int val = 1;
    gray8_image *test = new gray8_image(5, 5);
    for (int i = 0; i < test->sx; i++) {
        for (int j = 0; j < test->sy; j++) {
            test->pixels[i * test->sy + j] = val;
            val += 1;
        }
    }
    for (int i = 0; i < test->length; i++) {
        std::cout << test->pixels[i] << ";";
    }
    gray8_image *test2 = new gray8_image(5, 5);
    for (int i = 0; i < test2->sx; i++) {
        for (int j = 0; j < test2->sy; j++) {
            test2->pixels[i * test2->sx + j] = 1;
            std::cout << test2->pixels[i * test2->sx + j] << " ";
        
        }
        std::cout << "\n";
    }

    gray8_image *conv = test->gray_convolution(test2);
    for (int i = 0; i < conv->sx; i++) {
        for (int j = 0; j < conv->sy; j++) {
            std::cout << conv->pixels[i * conv->sy + j] << " ";
        }
        std::cout << "\n";
    }

    for (int i = 0; i < conv->length; i++) {
        std::cout << conv->pixels[i] << ";";
    }*/
    char *filename = "img/b001.png";
    read_png_file(filename);
    std::cout << "image loaded\n";
    gray8_image *test = new gray8_image(height, width, row_pointers);
    std::cout << "grayscale image\n";
    /*gray8_image *res = compute_harris_response(test);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout.precision(17);
            std::cout << res->pixels[i * width + j] << " ";
        }
    }*/
    detect_harris_points(test, 30, 25, 0.1);
    /*for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            png_bytep pixel = &(row_pointers[i][j * 4]);
            std::cout << "Image Pixel (x = 4, y = 7) : RGB(" << +pixel[0] << ", " << +pixel[1] << ", " << +pixel[2] << ")" << "\n";
            std::cout << "Image Pixel (x = 4, y = 7) : transparance : " << +pixel[3] << "\n";   
        }
        std::cout << "\n";
    }*/
    return 0;
}