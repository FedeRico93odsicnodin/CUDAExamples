#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <math.h>
#include <cmath>
#include "../common.h"

void load_image(char* fname, int Nx, int Ny, float* img) {
	FILE* fp;

	fp = fopen(fname, "r");

	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++)
			fscanf(fp, "%f", &img[i * Ny + j]);
		fscanf(fp, "\n");
	}

	fclose(fp);
}
void save_image(char* fname, int Nx, int Ny, float* img) {
	FILE* fp;

	fp = fopen(fname, "w");

	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			fprintf(fp, "%10.3f", img[i * Ny + j]);
			fprintf(fp, "\n");
		}

		
	}
	fclose(fp);
}
void calculate_kernel(int kernel_size, float sigma, float* kernel) {
	int Nk2 = kernel_size * kernel_size;
	float x, y, center;

	center = (kernel_size - 1) / 2.0;

	for (int i = 0; i < Nk2; i++) {
		x = (float)(i % kernel_size) - center;
		y = (float)(i / kernel_size) - center;
		kernel[i] = -(1.0 / M_PI * pow(sigma, 4)) * (1.0 - 0.5 * (x * x + y * y) / (sigma * sigma)) * exp(-0.5 * (x * x + y * y) / (sigma * sigma));
	}
}
void conv_img_cpu(float* img, float* kernel, float* imgf, int Nx, int Ny, int kernel_size) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			float sum = 0;
			for (int ki = 0; ki < kernel_size; ki++) {
				for (int kj = 0; kj < kernel_size; kj++) {
					int ii, jj;
					ii = i - 1 + ki;
					jj = j - 1 + kj;
					if (ii < 0 || ii >= Nx)
						ii = i;
					if (jj < 0 || jj >= Ny)
						jj = j;
					sum += img[ii * Ny + jj] * kernel[ki * kernel_size + kj];
				}
			}
			imgf[i * Ny + j] = sum;
		}
	}
}
/// <summary>
/// First implementation of the convolution image by the kernel function 
/// </summary>
/// <param name="img"></param>
/// <param name="kernel"></param>
/// <param name="imgf"></param>
/// <param name="Nx"></param>
/// <param name="Ny"></param>
/// <param name="kernel_size"></param>
/// <returns></returns>
__global__ void conv_img_gpu(float* img, float* kernel, float* imgf, int Nx, int Ny, int kernel_size) {
	int tid = threadIdx.x;
	int row = blockIdx.x;

	int K2 = kernel_size * kernel_size;

	extern __shared__ float sdata[];

	if (tid < K2)
		sdata[tid] = kernel[tid];

	__syncthreads();

	int column = tid;
	int idx = row * Ny + column;

	if (column < Ny) // check if we are in the boundaries of the original image 
	{
		float sum = 0;
		for (int ki = 0; ki < kernel_size; ki++) {
			for (int kj = 0; kj < kernel_size; kj++) {
				int ii, jj;
				ii = row - 1 + ki;
				jj = column - 1 + kj;
				if (ii < 0 || ii >= Nx)
					ii = row;
				if (jj < 0 || jj >= Ny)
					jj = column;
				sum += img[ii * Ny + jj] * kernel[ki * kernel_size + kj];
			}
		}
		imgf[idx] = sum;
	}
	__syncthreads();
}
void convolution() {
	int Nx, Ny;
	int kernel_size;
	float sigma;
	char finput[256], foutput[256];
	double start_cpu, end_cpu, start_gpu, end_gpu;

	// input for the file lux_bw
	printf("please insert for the lux_bw file location:");
	scanf(" %[^\n]%*c", &finput);
	// input for the file lux_output_hw
	printf("please insert for the lux_output_hw file location:");
	scanf(" %[^\n]%*c", &foutput);

	/*sprintf(finput, "lux_bw.bat");
	sprintf(foutput, "lux_output_hw.dat");*/

	// Image dimensions
	Nx = 570;
	Ny = 600;
	// Edge of the convolution 
	kernel_size = 32;
	sigma = 0.8;

	// allocate memory 
	float* img, * imgf, * imgf_hw, * kernel;
	img = (float*)malloc(Nx * Ny * sizeof(float));
	imgf = (float*)malloc(Nx * Ny * sizeof(float));
	imgf_hw = (float*)malloc(Nx * Ny * sizeof(float));
	kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

	load_image(finput, Nx, Ny, img);
	calculate_kernel(kernel_size, sigma, kernel);

	start_cpu = seconds();
	conv_img_cpu(img, kernel, imgf, Nx, Ny, kernel_size);
	end_cpu = seconds();

	printf("CPU time %lf\n", end_cpu - start_cpu);

	// GPU implementation 
	float* d_img, * d_imgf, * d_kernel;

	CHECK(cudaMalloc(&d_img, Nx * Ny * sizeof(float)));
	CHECK(cudaMalloc(&d_imgf, Nx * Ny * sizeof(float)));
	CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

	CHECK(cudaMemcpy(d_img, img, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(Nx, 1, 1); // one block per image row 
	dim3 threadsPerBlock(Ny, 1, 1); // one thread per image col

	start_gpu = seconds();
	conv_img_gpu << <blocksPerGrid, threadsPerBlock, kernel_size* kernel_size * sizeof(float) >> > (d_img, d_kernel, d_imgf, Nx, Ny, kernel_size);
	cudaDeviceSynchronize();
	end_gpu = seconds();
	CHECK(cudaMemcpy(imgf_hw, d_imgf, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost));
	save_image(foutput, Nx, Ny, imgf_hw);

	int check = 1;
	for (size_t i = 0; i < Nx; i++) {
		if (check) {
			for (size_t j = 0; j < Ny; j++) {
				if (abs(imgf[i * Ny + j] - imgf_hw[i * Ny + j]) > 0.0001) {
					printf("GPU convolution error\n");
					check = 0;
					break;
				}
			}
		}
		else
			break;
	}

	printf("GPU Time: %lf\n", end_gpu - start_gpu);

	free(img);
	free(imgf);
	free(imgf_hw);
	free(kernel);

	CHECK(cudaFree(d_img));
	CHECK(cudaFree(d_imgf));
	CHECK(cudaFree(d_kernel));
}
/// <summary>
/// Second implementation of the image convolution done by the kernel 
/// </summary>
/// <param name="img"></param>
/// <param name="kernel"></param>
/// <param name="imgf"></param>
/// <param name="Nx"></param>
/// <param name="Ny"></param>
/// <param name="kernel_size"></param>
/// <returns></returns>
__global__ void conv_img_gpu_allimagesupport(float* img, float* kernel, float* imgf, int Nx, int Ny, int kernel_size) {
	int tid = threadIdx.x;
	int row = blockIdx.x;

	int K2 = kernel_size * kernel_size;

	extern __shared__ float sdata[];

	if (tid < K2)
		sdata[tid] = kernel[tid];

	__syncthreads();

	for (int i = 0; i < Ny; i += blockDim.x) {
		int column = tid + i;
		// check if we are in the boundaries of the original image 
		if (column < Ny) {
			int idx = row * Ny + column;

			float sum = 0;
			for (int ki = 0; ki < kernel_size; ki++) {
				for (int kj = 0; kj < kernel_size; kj++) {
					int ii, jj;
					ii = row - 1 + ki;
					jj = column - 1 + kj;
					if (ii < 0 || ii >= Nx)
						ii = row;
					if (jj < 0 || jj >= Ny)
						jj = column;
					sum += img[ii * Ny + jj] * kernel[ki * kernel_size + kj];
				}
			}
			imgf[idx] = sum;
		}
	}
	__syncthreads();
}
void convolution_allimagesupport() {
	int Nx, Ny;
	int kernel_size;
	int BLOCKSIZE = 512;
	float sigma;
	char finput[256], foutput[256];
	double start_cpu, end_cpu, start_gpu, end_gpu;

	// input for the file lux_bw
	printf("please insert for the lux_bw file location:");
	scanf(" %[^\n]%*c", &finput);
	// input for the file lux_output_hw
	printf("please insert for the lux_output_hw file location:");
	scanf(" %[^\n]%*c", &foutput);
	// input for the blocksize
	printf("please enter the desired blocksize:");
	scanf(" %d", &BLOCKSIZE);

	// Image dimensions
	Nx = 570;
	Ny = 600;
	// Edge of the convolution
	kernel_size = 32;
	sigma = 0.8;

	// allocate memory 
	float* img, * imgf, * imgf_hw, * kernel;
	img = (float*)malloc(Nx * Ny * sizeof(float));
	imgf = (float*)malloc(Nx * Ny * sizeof(float));
	imgf_hw = (float*)malloc(Nx * Ny * sizeof(float));
	kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

	// loading the image and kernel calculus 
	load_image(finput, Nx, Ny, img);
	calculate_kernel(kernel_size, sigma, kernel);

	start_cpu = seconds();
	conv_img_cpu(img, kernel, imgf, Nx, Ny, kernel_size);
	end_cpu = seconds();

	// GPU Implementation
	float* d_img, * d_imgf, * d_kernel;

	CHECK(cudaMalloc(&d_img, Nx * Ny * sizeof(float)));
	CHECK(cudaMalloc(&d_imgf, Nx * Ny * sizeof(float)));
	CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

	CHECK(cudaMemcpy(d_img, img, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(Nx, 1, 1);               // One block per image row
	dim3 threadsPerBlock(BLOCKSIZE, 1, 1);      // Set by user input

	start_gpu = seconds();
	conv_img_gpu_allimagesupport << <blocksPerGrid, threadsPerBlock, kernel_size* kernel_size * sizeof(float) >> > (d_img, d_kernel, d_imgf, Nx, Ny, kernel_size);
	CHECK_KERNELCALL();
	cudaDeviceSynchronize();
	end_gpu = seconds();
	CHECK(cudaMemcpy(imgf_hw, d_imgf, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost));
	save_image(foutput, Nx, Ny, imgf_hw);

	int check = 1;
	for (size_t i = 0; i < Nx; i++)
	{
		if (check) {
			for (size_t j = 0; j < Ny; j++)
			{
				if (abs(imgf[i * Ny + j] - imgf_hw[i * Ny + j]) > 0.0001)
				{
					printf("GPU Convolution Error\n");
					check = 0;
					break;
				}
			}
		}
		else
			break;
	}

	printf("GPU Time: %lf\n", end_gpu - start_gpu);

	free(img);
	free(imgf);
	free(imgf_hw);
	free(kernel);

	CHECK(cudaFree(d_img));
	CHECK(cudaFree(d_imgf));
	CHECK(cudaFree(d_kernel));

}