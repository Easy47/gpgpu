#include <iostream> 
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <vector>
#include "image.hh"
#include "harris_gpu.hh"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>

using namespace ImageGpu;

namespace gpu {
__host__ void gauss_derivative_kernels(int size, int sizey, gray8_image *gx, gray8_image *gy) {
	
    double gx_tmp[49] = { 
	0.000308397, 0.00263511, 0.00608749, -0, -0.00608749, -0.00263511, -0.000308397,
	0.00395267, 0.0337738, 0.0780223, -0, -0.0780223, -0.0337738, -0.00395267,
	0.0182625, 0.156045, 0.360485, -0, -0.360485, -0.156045, -0.0182625,
	0.0304169, 0.259899, 0.600404, -0, -0.600404, -0.259899, -0.0304169,
	0.0182625, 0.156045, 0.360485, -0, -0.360485, -0.156045, -0.0182625,
	0.00395267, 0.0337738, 0.0780223, -0, -0.0780223, -0.0337738, -0.00395267,
	0.000308397, 0.00263511, 0.00608749, -0, -0.00608749, -0.00263511, -0.000308397,
    };

    double gy_tmp[49] = { 
	0.000308397, 0.00395267, 0.0182625, 0.0304169, 0.0182625, 0.00395267, 0.000308397, 
	0.00263511, 0.0337738, 0.156045, 0.259899, 0.156045, 0.0337738, 0.00263511, 
	0.00608749, 0.0780223, 0.360485, 0.600404, 0.360485, 0.0780223, 0.00608749, 
	-0, -0, -0, -0, -0, -0, -0, 
	-0.00608749, -0.0780223, -0.360485, -0.600404, -0.360485, -0.0780223, -0.00608749, 
	-0.00263511, -0.0337738, -0.156045, -0.259899, -0.156045, -0.0337738, -0.00263511, 
	-0.000308397, -0.00395267, -0.0182625, -0.0304169, -0.0182625, -0.00395267, -0.000308397, 
    };

    gx->get_data_from(gx_tmp);
    gy->get_data_from(gy_tmp);
}

__host__ void gauss_kernel(int size, int sizey, gray8_image *res) {

    double g_tmp[49] = { 
	0.000102799, 0.00131756, 0.00608749, 0.010139, 0.00608749, 0.00131756, 0.000102799, 
	0.00131756, 0.0168869, 0.0780223, 0.12995, 0.0780223, 0.0168869, 0.00131756, 
	0.00608749, 0.0780223, 0.360485, 0.600404, 0.360485, 0.0780223, 0.00608749, 
	0.010139, 0.12995, 0.600404, 1, 0.600404, 0.12995, 0.010139, 
	0.00608749, 0.0780223, 0.360485, 0.600404, 0.360485, 0.0780223, 0.00608749, 
	0.00131756, 0.0168869, 0.0780223, 0.12995, 0.0780223, 0.0168869, 0.00131756, 
	0.000102799, 0.00131756, 0.00608749, 0.010139, 0.00608749, 0.00131756, 0.000102799, 
    };
    res->get_data_from(g_tmp);
}

void print_img(gray8_image *img){
    std::cout << "{\n";
    for (int x = 0; x < img->sx; x++) {
	for (int y = 0; y < img->sy; y++) {
	    std::cout << img->pixels[x * img->sy + y] << ", ";
	}
	std::cout << "\n";
    }
    std::cout << "}\n";
}

void gauss_derivatives(gray8_image *img, int size, gray8_image *imx, gray8_image *imy) {

    gray8_image *gx = new gray8_image(2*size + 1, 2*size + 1);
    gray8_image *gy = new gray8_image(2*size + 1, 2*size + 1);
    gauss_derivative_kernels(size, size, gx, gy);


    dim3 dimBlockConvol(32, 32);
    dim3 dimGridConvol((imx->sx + dimBlockConvol.x - 1)/dimBlockConvol.x, (imx->sy + dimBlockConvol.y - 1)/dimBlockConvol.y);

    
    kvecConvol<<<dimBlockConvol,dimGridConvol>>>(img->pixels, img->sx, img->sy, gx->pixels, gx->sx, imx->pixels); 
    //img->gray_convolution(gx, imx);
    kvecConvol<<<dimBlockConvol,dimGridConvol>>>(img->pixels, img->sx, img->sy, gy->pixels, gy->sx, imy->pixels); 
    //img->gray_convolution(gy, imy);
    cudaDeviceSynchronize();


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

    gray8_image *imx2 = new gray8_image(imx->sx, imx->sy);
    gray8_image *Wxx = new gray8_image(imx2->sx, imx2->sy);

    gray8_image *imximy = new gray8_image(imx->sx, imx->sy);
    auto Wxy = new gray8_image(imximy->sx, imximy->sy);

    gray8_image *imy2 = new gray8_image(imx->sx, imx->sy);
    auto Wyy = new gray8_image(imy2->sx, imy2->sy);
	
    int dimBlock = (1024);
    int dimGrid = ((imx->length + dimBlock - 1)/dimBlock);
    dim3 dimBlockConvol(32, 32);
    dim3 dimGridConvol((imx->sx + dimBlockConvol.x - 1)/dimBlockConvol.x, (imx->sy + dimBlockConvol.y - 1)/dimBlockConvol.y);

    gray8_image *gauss1 = new gray8_image(2*OPENING_SIZE + 1, 2*OPENING_SIZE + 1);
    gauss_kernel(OPENING_SIZE, OPENING_SIZE, gauss1);
    gray8_image *gauss2 = new gray8_image(2*OPENING_SIZE + 1, 2*OPENING_SIZE + 1);
    gauss_kernel(OPENING_SIZE, OPENING_SIZE, gauss2);

    gray8_image *imx_stream = new gray8_image(img->sx, img->sy);
    gray8_image *imy_stream = new gray8_image(img->sx, img->sy);
    gauss_derivatives(img, DERIVATIVE_KERNEL_SIZE, imx_stream, imy_stream);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate ( &stream1);
    cudaStreamCreate ( &stream2);
    cudaStreamCreate ( &stream3);

    kvecMult<<<dimBlock,dimGrid, 0, stream3>>>(imx->pixels, imx->pixels, imx2->pixels, imx->length); 
    kvecMult<<<dimBlock,dimGrid, 0, stream1>>>(imx_stream->pixels, imy->pixels, imximy->pixels, imx->length);  
    kvecMult<<<dimBlock,dimGrid, 0, stream2>>>(imy_stream->pixels, imy_stream->pixels, imy2->pixels, imx->length); 

    kvecConvol<<<dimBlockConvol,dimGridConvol, 0, stream3>>>(imx2->pixels, imx2->sx, imx2->sy, gauss->pixels, gauss->sx, Wxx->pixels); 

    kvecConvol<<<dimBlockConvol,dimGridConvol, 0, stream1>>>(imximy->pixels, imximy->sx, imximy->sy, gauss1->pixels, gauss1->sx, Wxy->pixels); 

    kvecConvol<<<dimBlockConvol,dimGridConvol, 0, stream2>>>(imy2->pixels, imy2->sx, imy2->sy, gauss2->pixels, gauss2->sx, Wyy->pixels); 

    cudaDeviceSynchronize();

    gray8_image *s1 = new gray8_image(imx->sx, imx->sy);
    //dim3 dimGrid((imx->sx + dimBlock.x - 1)/dimBlock.x, (imx->sy + dimBlock.y - 1)/dimBlock.y);
    kvecMult<<<dimBlock,dimGrid>>>(Wxx->pixels, Wyy->pixels, s1->pixels, imx->length); 
    cudaDeviceSynchronize();
    //auto s1 = img_mult(Wxx, Wyy);


    gray8_image *s2 = new gray8_image(imx->sx, imx->sy);
    //dim3 dimGrid((imx->sx + dimBlock.x - 1)/dimBlock.x, (imx->sy + dimBlock.y - 1)/dimBlock.y);
    kvecMult<<<dimBlock,dimGrid>>>(Wxy->pixels, Wxy->pixels, s2->pixels, imx->length); 
    cudaDeviceSynchronize();
    //auto s2 = img_mult(Wxy, Wxy);


    gray8_image *Wdet = new gray8_image(imx->sx, imx->sy);
    //dim3 dimGrid((imx->sx + dimBlock.x - 1)/dimBlock.x, (imx->sy + dimBlock.y - 1)/dimBlock.y);
    kvecSous<<<dimBlock,dimGrid>>>(s1->pixels, s2->pixels, Wdet->pixels, imx->length); 

    // ENLEVABLE ??? -----------------------------
    //cudaDeviceSynchronize();
    //auto Wdet = img_sous(s1, s2);

    delete s1;
    delete s2;

    //auto Wtr = new gray8_image(Wxx->sx, Wxx->sy);

    gray8_image *Wtr = new gray8_image(imx->sx, imx->sy);

    kvecAdd<<<dimBlock,dimGrid>>>(Wxx->pixels, Wyy->pixels, Wtr->pixels, imx->length); 
    cudaDeviceSynchronize();


    gray8_image *tmp = new gray8_image(imx->sx, imx->sy);
    //dim3 dimGrid((imx->sx + dimBlock.x - 1)/dimBlock.x, (imx->sy + dimBlock.y - 1)/dimBlock.y);
    kvecAddScalar<<<dimBlock,dimGrid>>>(Wtr->pixels, 1, tmp->pixels, imx->length); 
    cudaDeviceSynchronize();
    //gray8_image *tmp = img_add_scalar(Wtr, 1);

    gray8_image *res = new gray8_image(imx->sx, imx->sy);
    //dim3 dimGrid((imx->sx + dimBlock.x - 1)/dimBlock.x, (imx->sy + dimBlock.y - 1)/dimBlock.y);
    kvecDiv<<<dimBlock,dimGrid>>>(Wdet->pixels, tmp->pixels, res->pixels, imx->length); 
    cudaDeviceSynchronize();
    //gray8_image *res = img_div(Wdet, tmp);

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
/*class Point {
    public:
    int x;
    int y;
    float val;
    Point(int x, int y, float val) {
        this->x = x;
        this->y = y;
        this->val = val;
    }
};*/

/*
__device__ static float atomicMax(float *address, float val) {
	int *address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCas(address_as_i, assumed, 
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ static float atomicMin(float *address, float val) {
	int *address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCas(address_as_i, assumed, 
				__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
*/

__device__ int dev_count_d = 0;

typedef struct
{
    int x, y;
} Point;

__device__ int my_push_back(double *harris, Point *coord, double harris_val, int *coord_val, int length) {
	int insert_pt = atomicAdd(&dev_count_d, 1);
	if (insert_pt < length) {
		harris[insert_pt] = harris_val;
		coord[insert_pt].x = coord_val[0];
		coord[insert_pt].y = coord_val[1];
		return insert_pt;
	}
	return -1;
}

__global__ void kvecComputeMask(double max, double min, double *img_pix, double *harris_pix, int harris_sx, int harris_sy, double *harris_vals, Point *coord, float threshold) {
    float rtol = 0.0001;
    float atol = 0.0000001;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= harris_sx)
	return;
    if (y >= harris_sy)
	return;

	if ((abs(harris_pix[x * harris_sy + y] - img_pix[x * harris_sy + y]) <= atol * rtol * abs(img_pix[x * harris_sy + y])) 
	&& (harris_pix[x * harris_sy + y] > min + threshold * (max - min))) {
		// Point tmp = Point(i , j, harris_resp->pixels[i * harris_resp->sy + j]);
		int coord_val[2] = {x, y};
		my_push_back(harris_vals, coord, harris_pix[x * harris_sy + y], coord_val, harris_sx * harris_sy);
	}
}

Point* compute_mask(gray8_image *harris_resp, gray8_image *t2, float threshold, int &point_nb) {
    std::vector<Point> res;
	
	thrust::device_vector<double> vec((double*)t2->pixels, (double*)(t2->pixels + t2->length));
	double max = *thrust::max_element(vec.begin(), vec.end());
	double min = *thrust::min_element(vec.begin(), vec.end());

	double *harris_vals;
	Point *coord;
	int length = harris_resp->length;
    cudaMallocManaged(&harris_vals, sizeof(double) * length);
    cudaMallocManaged(&coord, sizeof(Point) * length);

    dim3 dimBlockConvol(32, 32);
    dim3 dimGridConvol((harris_resp->sx + dimBlockConvol.x - 1)/dimBlockConvol.x, (harris_resp->sy + dimBlockConvol.y - 1)/dimBlockConvol.y);

	kvecComputeMask<<<dimBlockConvol, dimGridConvol>>>(max, min, t2->pixels, harris_resp->pixels, harris_resp->sx, harris_resp->sy, harris_vals, coord, threshold);
	cudaDeviceSynchronize();

	int dev_count;
	cudaMemcpyFromSymbol(&dev_count, dev_count_d, sizeof(dev_count_d), 0, cudaMemcpyDeviceToHost);

	/*
    for (int i = 0; i < harris_resp->sx; i++) {
        for (int j = 0; j < harris_resp->sy; j++) {
            if ((abs(harris_resp->pixels[i * harris_resp->sy + j] - t2->pixels[i * harris_resp->sy + j]) <= atol * rtol * abs(t2->pixels[i * harris_resp->sy + j])) 
            && (harris_resp->pixels[i * harris_resp->sy + j] > min + threshold * (max - min))) {
                //mask->pixels[i * harris_resp->sy + j] = 1;
                Point tmp = Point(i , j, harris_resp->pixels[i * harris_resp->sy + j]);
                res.push_back(tmp);
            }
        }
    }
	*/

	
	// print coords
	/*for (int i = 0; i < dev_count; i++) {
		std::cout << coord[i].x << " " << coord[i].x << std::endl;
	}*/
	

	point_nb = dev_count;

	double *sorted_harris_vals;
	int **sorted_coord;
    cudaMallocManaged(&sorted_harris_vals, sizeof(double) * length);
    cudaMallocManaged(&sorted_coord, sizeof(int*) * length);

	/*void *d_tmp_storage = NULL;
	size_t tmp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(d_tmp_storage, tmp_storage_bytes, harris_vals, sorted_harris_vals, coord, sorted_coord, dev_count);

	cudaMallocManaged(&d_tmp_storage, tmp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(d_tmp_storage, tmp_storage_bytes, harris_vals, sorted_harris_vals, coord, sorted_coord, dev_count);

	cudaDeviceSynchronize();

	for (int i = 0; i < dev_count; i++) {
		std::cout << sorted_harris_vals[i] << std::endl;
	}

    // std::sort(candidate.begin(), candidate.end(), myfunction);
	//TODO fix coords allocation*/
	thrust::sort_by_key(harris_vals, harris_vals + dev_count, coord);
    return coord;
}

//bool myfunction (Point p1,Point p2) { return ( p1.val < p2.val); }

Point *detect_harris_points(gray8_image *image_gray, int max_keypoints = 30, int min_distance = 25, float threshold = 0.1) {

    gray8_image *harris_resp = compute_harris_response(image_gray);
    double tmp[625] = { 
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
	ellipse_kernel->get_data_from(tmp);

    gray8_image *dilate = new gray8_image(harris_resp->sx, harris_resp->sy);
    dim3 dimBlockConvol(32, 32);
    dim3 dimGridConvol((dilate->sx + dimBlockConvol.x - 1)/dimBlockConvol.x, (dilate->sy + dimBlockConvol.y - 1)/dimBlockConvol.y);
    
    kvecDilate<<<dimBlockConvol,dimGridConvol>>>(harris_resp->pixels, harris_resp->sx, harris_resp->sy, ellipse_kernel->pixels, ellipse_kernel->sx, dilate->pixels); 
    //gray8_image *dilate = harris_resp->dilate(ellipse_kernel);
    cudaDeviceSynchronize();

    //gray8_image *mask = new gray8_image(image_gray->sx, image_gray->sy);
	int point_nb;
    Point *coord = compute_mask(dilate, harris_resp, threshold, point_nb);
	/*
	for (int i = 0; i < 30; i++) {
		std::cout << coord[i].x << " " << coord[i].y << std::endl;
	}
	*/

	int length = point_nb < max_keypoints ? point_nb : max_keypoints;
	Point *res = (Point*) malloc(sizeof(Point) * length);
	cudaMemcpy(res, coord, sizeof(Point) * length, cudaMemcpyDeviceToHost);
	/*
    int nb = 0;
    for (auto i = candidate.begin(); i != candidate.end(); i++) {
        if (nb == max_keypoints) {
            break;
        }
        nb++;
        res.push_back(*i);
    }
	*/
    delete harris_resp;
    delete ellipse_kernel;
    delete dilate;
    return res;
}

void detect_point(PNG_data image_data) {
    //std::cout << "started" << std::endl;
    gray8_image *test = new gray8_image(image_data.height, image_data.width, image_data.row_pointers);
    //std::cout << "grayscale image\n";
    Point *res = detect_harris_points(test, 30, 25, 0.1);
    //std::cout << res.size();
	/*
    for (auto i = res.begin(); i != res.end(); i++) {
        std::cout << i.x << " " << i.y << std::endl;
    }
	*/
    delete test;
}
}
