#ifndef IP
#define IP
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

#define NO_CHANNELS 3
#define M_PI 3.14159265358979323846

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define RADIUS 5            // Maximum kernel size is 11
#define MAX_WINDOW_SIZE 121            // Maximum kernel size is 11

enum Direction {DIR_X, DIR_Y};

int divUp(int a, int b);


// Kernels
void startCUDA_Convolution_2D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst);

void startCUDA_Convolution_1D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst, Direction type);

void startCUDA_Denoise(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float percentage);


/************************************Image******************************************************/
class Image {
	public:
		Image();
		Image(std::string filename);
        Image(cv::Mat img);
		~Image();

        Image operator=(const Image &img);

        Image Blur(int kernel_size, float sigma);
        Image laplacian_Filter();
        Image gaussian_Separable(int kernel_size, float sigma);
        Image denoise(int kernel_size, float percentage);

        // Display Image
		void display(std::string text);

    private:
        // Initialization Method
		void init(std::string filename);

        cv::cuda::GpuMat get_2D_GaussianKernel(int kernel_size, float sigma) const; // creating gaussian kernel for GPU Memory
        cv::cuda::GpuMat get_1D_GaussianKernel(int kernel_size, float sigma) const; // creating gaussian kernel for GPU Memory

        cv::Mat host_image;
        cv::cuda::GpuMat device_image;
};

#endif