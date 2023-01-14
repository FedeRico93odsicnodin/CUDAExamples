#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "../common.h"
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <errno.h>

#define MAX_LENGTH 614400
// sequential method for calculation of the histogram
void sequential_histogram(char* data, unsigned int* histogram, int length) {
	for (int i = 0; i < length; i++) {
		int alphabet_position = data[i] - 'a';
		// check if we have an alphabet char 
		if (alphabet_position >= 0 && alphabet_position < 26) {
			// grouping the letters into blocks of 6
			histogram[alphabet_position / 6] ++;
		}
	}
}
// kernel implementation for the calcualation of the histogram
__global__ void histogram_kernel(char* data, unsigned int* histogram, int length) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int section_size = (length - 1) / (blockDim.x * gridDim.x) + 1;
	int start = i * section_size;
	// all threads handle blockDim.x * gridDim.x consecutive elements 
	for (size_t k = 0; k < section_size; k++) {
		if (start + k < length) {
			int alphabet_position = data[start + k] - 'a';
			if (alphabet_position >= 0 && alphabet_position < 26) {
				atomicAdd((&histogram[alphabet_position / 6]), 1);
			}
		}
	}
}
void histogramGPU() {
	int maxLen = MAX_LENGTH;
	// input file to which executing the histograms 
	printf("please insert the input file parameter:");
	char finput[200];
	scanf(" %[^\n]%*c", &finput);
	FILE* fp = fopen(finput, "r");
	printf("result on file open: %s\n", strerror(errno));
	if (fp == NULL)
		return;
	int BLOCKDIM = 256;
	printf("please insert the block dimension:");
	scanf("%d", &BLOCKDIM);
	// unsigned char text[MAX_LENGTH]
	char* text = (char*)malloc(sizeof(char) * MAX_LENGTH);
	char* text_d;
	char* read;
	int len = 0;
	unsigned int histogram[5] = { 0 };
	unsigned int histogram_hw[5] = { 0 };
	unsigned int* histogram_d;
	double start_cpu, end_cpu, start_gpu, end_gpu;
	
	
	while ((read = fgets(text, maxLen, fp)) != NULL) {
		len = strlen(text);
		printf("retrieved line of length: %d\n", len);
	}
	fclose(fp);

	start_cpu = seconds();
	sequential_histogram(text, histogram, len);
	end_cpu = seconds();

	// allocate space for the input array on the GPU
	CHECK(cudaMalloc(&text_d, len * sizeof(char)));
	// and for the histogram
	CHECK(cudaMalloc(&histogram_d, 5 * sizeof(unsigned int)));
	// copy input data on the gpu
	CHECK(cudaMemcpy(text_d, text, len * sizeof(char), cudaMemcpyHostToDevice));

	dim3 blocksPerGrid((maxLen + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);
	start_gpu = seconds();
	histogram_kernel << <blocksPerGrid, threadsPerBlock >> > (text_d, histogram_d, len);
	CHECK_KERNELCALL();

	cudaDeviceSynchronize();
	end_gpu = seconds();
	// copy data back from the gpu
	CHECK(cudaMemcpy(histogram_hw, histogram_d, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printf("\n");
	for (size_t i = 0; i < 5; i++) {
		printf("CPU: section %d - count %d\n", i, histogram[i]);
	}
	for (size_t i = 0; i < 5; i++) {
		printf("GPU: section %d - count %d\n", i, histogram_hw[i]);
	}
	printf("\n");
	printf("\n");
	for (size_t i = 0; i < 5; i++) {
		if (histogram[i] != histogram_hw[i]) {
			printf("Error on GPU at index: %ld\n", i);
			return;
		}
	}
	printf("ALL GPU OK\n");

	printf("CPU Sort Time %.5lf\n", end_cpu - start_cpu);
	printf("GPU Sort Time %.5lf\n", end_gpu - start_gpu);

	CHECK(cudaFree(text_d));
	CHECK(cudaFree(histogram_d));


}
/// <summary>
/// second version for the function on the kernel 
/// this is the coalesced version
/// </summary>
/// <param name="data"></param>
/// <param name="histogram"></param>
/// <param name="length"></param>
/// <returns></returns>
__global__ void histogram_kernel_v2(char* data, unsigned int* histogram, int length) {
	unsigned int tid = threadIdx.x + blockDim.x * blockDim.x;
	// all threads handle blockDim.x * gridDim.x consecutive elements in each iteration 
	for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26)
			atomicAdd(&(histogram[alphabet_position / 6]), 1);
	}
}
void histogramGPUCoalesced() {
	int maxLen = MAX_LENGTH;
	// input file to which executing the histograms 
	printf("please insert the input file parameter:");
	char finput[200];
	scanf(" %[^\n]%*c", &finput);
	FILE* fp = fopen(finput, "r");
	printf("result on file open: %s\n", strerror(errno));
	if (fp == NULL)
		return;
	int BLOCKDIM = 256;
	printf("please insert the block dimension:");
	scanf("%d", &BLOCKDIM);

	char* text = (char*)malloc(sizeof(char) * MAX_LENGTH);
	char* text_d;
	size_t len = 0;
	char* read;
	unsigned int histogram[5] = { 0 };
	unsigned int histogram_hw[5] = { 0 };
	unsigned int* histogram_d;
	double start_cpu, end_cpu, start_gpu, end_gpu;

	

	while ((read = fgets(text, maxLen, fp)) != NULL) {
		len = strlen(text);
		printf("retrieved line of length: %d\n", len);
	}
	fclose(fp);

	start_cpu = seconds();
	sequential_histogram(text, histogram, len);
	end_cpu = seconds();

	CHECK(cudaMalloc(&text_d, len * sizeof(char)));                              // allocate space for the input array on the GPU
	CHECK(cudaMalloc(&histogram_d, 5 * sizeof(unsigned int)));                   // and for the histogram
	CHECK(cudaMemcpy(text_d, text, len * sizeof(char), cudaMemcpyHostToDevice)); // copy input data on the gpu

	dim3 blocksPerGrid((len + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);
	start_gpu = seconds();
	histogram_kernel_v2 << <blocksPerGrid, threadsPerBlock >> > (text_d, histogram_d, len);
	CHECK_KERNELCALL();

	cudaDeviceSynchronize();
	end_gpu = seconds();

	CHECK(cudaMemcpy(histogram_hw, histogram_d, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost)); // copy data back from the gpu
	// printing the sections for cpu and gpu
	printf("\n");
	for (size_t i = 0; i < 5; i++) {
		printf("CPU: section %d - count %d\n", i, histogram[i]);
	}
	for (size_t i = 0; i < 5; i++) {
		printf("GPU: section %d - count %d\n", i, histogram_hw[i]);
	}
	printf("\n");
	printf("\n");
	for (size_t i = 0; i < 5; i++)
	{
		if (histogram[i] != histogram_hw[i])
		{
			printf("Error on GPU at index: %ld\n", i);
			return;
		}
	}
	printf("ALL GPU OK\n");

	printf("CPU Sort Time: %.5lf\n", end_cpu - start_cpu);
	printf("GPU Sort Time: %.5lf\n", end_gpu - start_gpu);

	CHECK(cudaFree(text_d));
	CHECK(cudaFree(histogram_d));

}
/// <summary>
/// Third version of the kernel histogram with the definition of bins 
/// </summary>
/// <param name="data"></param>
/// <param name="histogram"></param>
/// <param name="length"></param>
/// <param name="num_bins"></param>
/// <returns></returns>
__global__ void histogram_kernel_v3(char* data, unsigned int* histogram, unsigned int length, unsigned int num_bins) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// privatized bins 
	extern __shared__ unsigned int histo_s[];
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		if (binIdx < num_bins)
			histo_s[binIdx] = 0u;
	}
	__syncthreads();
	// histogram 
	for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26)
			atomicAdd(&(histo_s[alphabet_position / 6]), 1);
	}
	__syncthreads();
	// commit to global memory 
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		atomicAdd(&(histogram[binIdx]), histo_s[binIdx]);
	}
}
void histogramGPUCoalescedPrivate() {
	int maxLen = MAX_LENGTH;
	// input file to which executing the histograms 
	printf("please insert the input file parameter:");
	char finput[200];
	scanf(" %[^\n]%*c", &finput);
	FILE* fp = fopen(finput, "r");
	printf("result on file open: %s\n", strerror(errno));
	if (fp == NULL)
		return;
	int BLOCKDIM = 256;
	printf("please insert the block dimension:");
	scanf("%d", &BLOCKDIM);

	// unsigned char text[MAX_LENGTH];
	char* text = (char*)malloc(sizeof(char) * MAX_LENGTH);
	char* text_d;
	size_t len = 0;
	char* read;
	unsigned int histogram[5] = { 0 };
	unsigned int histogram_hw[5] = { 0 };
	unsigned int* histogram_d;
	double start_cpu, end_cpu, start_gpu, end_gpu;

	

	while ((read = fgets(text, maxLen, fp)) != NULL) {
		len = strlen(text);
		printf("retrieved line of length: %d\n", len);
	}
	fclose(fp);

	start_cpu = seconds();
	sequential_histogram(text, histogram, len);
	end_cpu = seconds();

	CHECK(cudaMalloc(&text_d, len * sizeof(char)));                              // allocate space for the input array on the GPU
	CHECK(cudaMalloc(&histogram_d, 5 * sizeof(unsigned int)));                   // and for the histogram
	CHECK(cudaMemcpy(text_d, text, len * sizeof(char), cudaMemcpyHostToDevice)); // copy input data on the gpu

	dim3 blocksPerGrid((len + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);
	start_gpu = seconds();
	histogram_kernel_v3 << <blocksPerGrid, threadsPerBlock, sizeof(unsigned int) * 5 >> > (text_d, histogram_d, len, 5); //same nof as the regular histogram
	CHECK_KERNELCALL();

	cudaDeviceSynchronize();
	end_gpu = seconds();
	CHECK(cudaMemcpy(histogram_hw, histogram_d, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost)); // copy data back from the gpu
	for (size_t i = 0; i < 5; i++)
	{
		if (histogram[i] != histogram_hw[i])
		{
			printf("Error on GPU at index: %ld\n", i);
			return;
		}
	}

	printf("ALL GPU OK\n");

	printf("CPU Sort Time: %.5lf\n", end_cpu - start_cpu);
	printf("GPU Sort Time: %.5lf\n", end_gpu - start_gpu);

	CHECK(cudaFree(text_d));
	CHECK(cudaFree(histogram_d));

}