#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "../include/IP.h"

int divUp(int a, int b){
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// 2D Convolution Operator
__global__ void Convolution_2D(const cv::cuda::PtrStep<uchar3> src, const cv::cuda::PtrStep<float> kernel, cv::cuda::PtrStep<uchar3> dst,
                                    int rows, int cols, int kernel_size){

  // Creating Shared Memory
  __shared__ uchar3 temp[BLOCKDIM_Y + (2 * RADIUS)][BLOCKDIM_X + (2 * RADIUS)];

   // Local Index
  int lindex_x = threadIdx.x + RADIUS;
  int lindex_y = threadIdx.y + RADIUS;

  // Global Index
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x >= 0 && dst_x < cols && dst_y >= 0 && dst_y < rows){
    // Read input elements into shared memory
    temp[lindex_y][lindex_x] = src(dst_y, dst_x);

    if(threadIdx.x < RADIUS){
      temp[lindex_y][lindex_x - RADIUS] = (dst_x - RADIUS >= 0)? src(dst_y, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on left-side of block
      temp[lindex_y][lindex_x + BLOCKDIM_X] = (dst_x + BLOCKDIM_X < cols)? src(dst_y, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on right-side of block
    }

    if(threadIdx.y < RADIUS){
      temp[lindex_y - RADIUS][lindex_x] = (dst_y - RADIUS >= 0)? src(dst_y - RADIUS, dst_x): make_uchar3(0, 0, 0); // Pixels on upper-side of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x] = (dst_y + BLOCKDIM_Y < rows)? src(dst_y + BLOCKDIM_Y, dst_x): make_uchar3(0, 0, 0); // Pixels on lower-side of block
    }

    if(threadIdx.x < RADIUS && threadIdx.y < RADIUS){
      temp[lindex_y - RADIUS][lindex_x - RADIUS] = ((dst_y - RADIUS >= 0) && (dst_x - RADIUS >= 0))? src(dst_y - RADIUS, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on top-left corner of block
      temp[lindex_y - RADIUS][lindex_x + BLOCKDIM_X] = ((dst_y - RADIUS >= 0) && (dst_x + BLOCKDIM_X < cols))? src(dst_y - RADIUS, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on top-right corner of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x - RADIUS] = ((dst_y + BLOCKDIM_Y < rows) && (dst_x - RADIUS >= 0))? src(dst_y + BLOCKDIM_Y, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on bottom-left corner of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x + BLOCKDIM_X] = ((dst_y + BLOCKDIM_Y < rows) && (dst_x + BLOCKDIM_X < cols))? src(dst_y + BLOCKDIM_Y, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on bottom-right corner of block
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    float pixel_sum[NO_CHANNELS] = {0.0};

    const int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    int size = (k_size - 1) / 2;

    for (int k = -size; k <= size; k++) {
      for (int l = -size; l <= size; l++) {
        pixel_sum[0] += (float)temp[lindex_y + k][lindex_x + l].x * kernel(k + size, l + size);
        pixel_sum[1] += (float)temp[lindex_y + k][lindex_x + l].y * kernel(k + size, l + size);
        pixel_sum[2] += (float)temp[lindex_y + k][lindex_x + l].z * kernel(k + size, l + size);
      }
    }

    dst(dst_y, dst_x).x = (unsigned char)pixel_sum[0];
    dst(dst_y, dst_x).y = (unsigned char)pixel_sum[1];
    dst(dst_y, dst_x).z = (unsigned char)pixel_sum[2];

  }
}

void startCUDA_Convolution_2D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  Convolution_2D<<<numBlocks, threadsPerBlock>>>(src, kernel, dst, dst.rows, dst.cols, kernel.rows);
}

// 1D Convolution Operator: Along Y-axis
__global__ void Convolution_1D_Y(const cv::cuda::PtrStep<uchar3> src, const cv::cuda::PtrStep<float> kernel, cv::cuda::PtrStep<uchar3> dst,int rows, int cols, int kernel_size){

  // Creating Shared Memory
  __shared__ uchar3 temp[BLOCKDIM_Y + (2 * RADIUS)][BLOCKDIM_X];

   // Local Index
  int lindex_x = threadIdx.x;
  int lindex_y = threadIdx.y + RADIUS;

  // Global Index
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x >= 0 && dst_x < cols && dst_y >= 0 && dst_y < rows){
    // Read input elements into shared memory
    temp[lindex_y][lindex_x] = src(dst_y, dst_x);

    if(threadIdx.y < RADIUS){
      temp[lindex_y - RADIUS][lindex_x] = (dst_y - RADIUS >= 0)? src(dst_y - RADIUS, dst_x): make_uchar3(0, 0, 0); // Pixels on upper-side of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x] = (dst_y + BLOCKDIM_Y < rows)? src(dst_y + BLOCKDIM_Y, dst_x): make_uchar3(0, 0, 0); // Pixels on lower-side of block
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    float pixel_sum[NO_CHANNELS] = {0.0};

    const int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    int size = (k_size - 1) / 2;

    for (int k = -size; k <= size; k++) {
      pixel_sum[0] += (float)temp[lindex_y + k][lindex_x].x * kernel(0, k + size);
      pixel_sum[1] += (float)temp[lindex_y + k][lindex_x].y * kernel(0, k + size);
      pixel_sum[2] += (float)temp[lindex_y + k][lindex_x].z * kernel(0, k + size);
    }
   
    dst(dst_y, dst_x).x = (unsigned char)pixel_sum[0];
    dst(dst_y, dst_x).y = (unsigned char)pixel_sum[1];
    dst(dst_y, dst_x).z = (unsigned char)pixel_sum[2];
  }
}

// 1D Convolution Operator: Along X-axis
__global__ void Convolution_1D_X(const cv::cuda::PtrStep<uchar3> src, const cv::cuda::PtrStep<float> kernel, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernel_size){

  // Creating Shared Memory
  __shared__ uchar3 temp[BLOCKDIM_Y][BLOCKDIM_X + (2 * RADIUS)];

   // Local Index
  int lindex_x = threadIdx.x + RADIUS;
  int lindex_y = threadIdx.y;

  // Global Index
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x >= 0 && dst_x < cols && dst_y >= 0 && dst_y < rows){
    // Read input elements into shared memory
    temp[lindex_y][lindex_x] = src(dst_y, dst_x);

    if(threadIdx.x < RADIUS){
      temp[lindex_y][lindex_x - RADIUS] = (dst_x - RADIUS >= 0)? src(dst_y, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on left-side of block
      temp[lindex_y][lindex_x + BLOCKDIM_X] = (dst_x + BLOCKDIM_X < cols)? src(dst_y, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on right-side of block
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    float pixel_sum[NO_CHANNELS] = {0.0};

    const int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    int size = (k_size - 1) / 2;

    for (int l = -size; l <= size; l++) {
      pixel_sum[0] += (float)temp[lindex_y][lindex_x + l].x * kernel(0, l + size);
      pixel_sum[1] += (float)temp[lindex_y][lindex_x + l].y * kernel(0, l + size);
      pixel_sum[2] += (float)temp[lindex_y][lindex_x + l].z * kernel(0, l + size);
    }
   
    dst(dst_y, dst_x).x = (unsigned char)pixel_sum[0];
    dst(dst_y, dst_x).y = (unsigned char)pixel_sum[1];
    dst(dst_y, dst_x).z = (unsigned char)pixel_sum[2];
  }
}


void startCUDA_Convolution_1D(cv::cuda::GpuMat& src, cv::cuda::GpuMat& kernel, cv::cuda::GpuMat& dst, Direction type){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));
  if(type == DIR_X)  Convolution_1D_X<<<numBlocks, threadsPerBlock>>>(src, kernel, dst, dst.rows, dst.cols, kernel.rows);
  else               Convolution_1D_Y<<<numBlocks, threadsPerBlock>>>(src, kernel, dst, dst.rows, dst.cols, kernel.rows);
}

// Denoise
__device__ void sort(float *array, int window_size){
	for (int i = 0; i < window_size - 1; i++) {
		for (int j = 0; j < window_size - i - 1; j++) {
			if (array[j] > array[j + 1]) {
				float temp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = temp;
			}
		}
	}
}

__global__ void Denoise(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernel_size, float percentage){

  // Creating Shared Memory
  __shared__ uchar3 temp[BLOCKDIM_Y + (2 * RADIUS)][BLOCKDIM_X + (2 * RADIUS)];

   // Local Index
  int lindex_x = threadIdx.x + RADIUS;
  int lindex_y = threadIdx.y + RADIUS;

  // Global Index
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x >= 0 && dst_x < cols && dst_y >= 0 && dst_y < rows){
    // Read input elements into shared memory
    temp[lindex_y][lindex_x] = src(dst_y, dst_x);

    if(threadIdx.x < RADIUS){
      temp[lindex_y][lindex_x - RADIUS] = (dst_x - RADIUS >= 0)? src(dst_y, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on left-side of block
      temp[lindex_y][lindex_x + BLOCKDIM_X] = (dst_x + BLOCKDIM_X < cols)? src(dst_y, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on right-side of block
    }

    if(threadIdx.y < RADIUS){
      temp[lindex_y - RADIUS][lindex_x] = (dst_y - RADIUS >= 0)? src(dst_y - RADIUS, dst_x): make_uchar3(0, 0, 0); // Pixels on upper-side of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x] = (dst_y + BLOCKDIM_Y < rows)? src(dst_y + BLOCKDIM_Y, dst_x): make_uchar3(0, 0, 0); // Pixels on lower-side of block
    }

    if(threadIdx.x < RADIUS && threadIdx.y < RADIUS){
      temp[lindex_y - RADIUS][lindex_x - RADIUS] = ((dst_y - RADIUS >= 0) && (dst_x - RADIUS >= 0))? src(dst_y - RADIUS, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on top-left corner of block
      temp[lindex_y - RADIUS][lindex_x + BLOCKDIM_X] = ((dst_y - RADIUS >= 0) && (dst_x + BLOCKDIM_X < cols))? src(dst_y - RADIUS, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on top-right corner of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x - RADIUS] = ((dst_y + BLOCKDIM_Y < rows) && (dst_x - RADIUS >= 0))? src(dst_y + BLOCKDIM_Y, dst_x - RADIUS): make_uchar3(0, 0, 0); // Pixels on bottom-left corner of block
      temp[lindex_y + BLOCKDIM_Y][lindex_x + BLOCKDIM_X] = ((dst_y + BLOCKDIM_Y < rows) && (dst_x + BLOCKDIM_X < cols))? src(dst_y + BLOCKDIM_Y, dst_x + BLOCKDIM_X): make_uchar3(0, 0, 0); // Pixels on bottom-right corner of block
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    const int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    const int window_size = k_size * k_size;
    int size = (k_size - 1) / 2;
    // Get array when window is convolved over image
    float R_array[MAX_WINDOW_SIZE];
    float G_array[MAX_WINDOW_SIZE];
    float B_array[MAX_WINDOW_SIZE];
    int counter = 0;

    for (int k = -size; k <= size; k++) {
      for (int l = -size; l <= size; l++) {
        B_array[counter] = (float)temp[lindex_y + k][lindex_x + l].x;
        G_array[counter] = (float)temp[lindex_y + k][lindex_x + l].y;
        R_array[counter] = (float)temp[lindex_y + k][lindex_x + l].z;
        counter++;
      }
    }

    // Sort neighbors (RGB arrays) in ascending order
    sort(B_array, window_size);
    sort(G_array, window_size);
    sort(R_array, window_size);

    // Get median - since window_size is always odd, index = window_size/2
    int index = int(window_size / 2);

    // Using either Median Filter or Average Filter
    int length = (window_size * percentage / 100) / 2.0;

    float r = 0.0, g = 0.0, b = 0.0;
    for (int i = index - length; i <= index + length; i++) {
      b += B_array[i];
      g += G_array[i];
      r += R_array[i];
    }
    dst(dst_y, dst_x).x = (unsigned char)(b / float(2 * length + 1));
    dst(dst_y, dst_x).y = (unsigned char)(g / float(2 * length + 1));
    dst(dst_y, dst_x).z = (unsigned char)(r / float(2 * length + 1));

  }
}

void startCUDA_Denoise(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float percentage){
  const dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  const dim3 numBlocks(divUp(dst.cols, threadsPerBlock.x), divUp(dst.rows, threadsPerBlock.y));

  Denoise<<<numBlocks, threadsPerBlock>>>(src, dst, dst.rows, dst.cols, kernel_size, percentage);
}