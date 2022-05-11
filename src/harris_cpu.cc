#include <iostream> 
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <vector>
#include "image.hh"
#include "harris_cpu.hpp"

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
    delete y;
    delete x;
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
    delete y;
    delete x;
}

void gauss_derivatives(gray8_image *img, int size, gray8_image *imx, gray8_image *imy) {

    gray8_image *gx = new gray8_image(2*size + 1, 2*size + 1);
    gray8_image *gy = new gray8_image(2*size + 1, 2*size + 1);
    gauss_derivative_kernels(size, size, gx, gy);

    
    img->gray_convolution(gx, imx);
    img->gray_convolution(gy, imy);


    delete gx;
    delete gy;
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

    gray8_image *Wxx = new gray8_image(imx2->sx, imx2->sy);
    imx2->gray_convolution(gauss, Wxx);

    gray8_image *imximy = img_mult(imx, imy);   

    auto Wxy = new gray8_image(imximy->sx, imximy->sy);
    imximy->gray_convolution(gauss, Wxy);

    gray8_image *imy2 = img_mult(imy, imy);

    auto Wyy = new gray8_image(imy2->sx, imy2->sy);
    imy2->gray_convolution(gauss, Wyy);

    auto s1 = img_mult(Wxx, Wyy);
    auto s2 = img_mult(Wxy, Wxy);
    auto Wdet = img_sous(s1, s2);
    delete s1;
    delete s2;

    auto Wtr = new gray8_image(Wxx->sx, Wxx->sy);
    img_add(Wxx, Wyy, Wtr);

    gray8_image *tmp = img_add_scalar(Wtr, 1);
    gray8_image *res = img_div(Wdet, tmp);

    delete imx;
    delete imy;
    delete gauss;
    delete imx2;
    delete imximy;
    delete imy2;
    delete tmp;



    delete Wxx;
    delete Wxy;
    delete Wyy;

    delete Wtr;
    delete Wdet;

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


 std::vector<Point> detect_harris_points(gray8_image *image_gray, int max_keypoints = 30, int min_distance = 25, float threshold = 0.1) {
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

    gray8_image *ellipse_kernel = new gray8_image(25,25);
    for (int i = 0; i < 625; i++) {
        ellipse_kernel->pixels[i] = tmp[i];
    }
    gray8_image *dilate = harris_resp->dilate(ellipse_kernel);

    //gray8_image *mask = new gray8_image(image_gray->sx, image_gray->sy);
    std::vector<Point> candidate = compute_mask(dilate, harris_resp, threshold);
    std::sort(candidate.begin(), candidate.end(), myfunction);

    std::vector<Point> res;
    int nb = 0;
    for (auto i = candidate.begin(); i != candidate.end(); i++) {
        if (nb == max_keypoints) {
            break;
        }
        nb++;
        res.push_back(*i);
    }
    delete harris_resp;
    delete ellipse_kernel;
    delete dilate;
    return res;
    
}
void detect_point(PNG_data image_data) {
    std::cout << "started" << std::endl;
    gray8_image *test = new gray8_image(image_data.height, image_data.width, image_data.row_pointers);
    std::cout << "grayscale image\n";
    /*gray8_image *res = compute_harris_response(test);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout.precision(17);
            std::cout << res->pixels[i * width + j] << " ";
        }
    }*/
    std::vector<Point> res = detect_harris_points(test, 30, 25, 0.1);
    for (auto i = res.begin(); i != res.end(); i++) {
        std::cout << (*i).x << " " << (*i).y << std::endl;
    }
    delete test;
}
