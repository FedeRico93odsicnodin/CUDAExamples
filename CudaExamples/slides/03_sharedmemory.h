#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common.h"

#define DIM_SH 204800;

void p_sum_cpu(float* p_sum, float* input, int length) // simple prefix sum in cpu
{
	p_sum[0] = input[0];
	int i;
	for (i = 1; i < length; ++i)
	{
		p_sum[i] = p_sum[i - 1] + input[i];
	}
}
__inline__ __device__ void warpReduce(volatile float* input, int threadId) {
	input[threadId] += input[threadId + 32];
	input[threadId] += input[threadId + 16];
	input[threadId] += input[threadId + 8];
	input[threadId] += input[threadId + 4];
	input[threadId] += input[threadId + 2];
	input[threadId] += input[threadId + 1];
}
// compute parallel reduction for each gpu block using multiple threads
__global__ void p_sum_gpu_shared(float* input) {
	unsigned threadId = threadIdx.x;
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float localVars[1024];

	localVars[threadId] = input[index];
	__syncthreads();

	// reduce like before until we only have 32 elements 
	for (unsigned i = blockDim.x / 2; i > 0; i >>= 1) {
		if (threadId < i) {
			localVars[threadId] += localVars[threadId + i];
		}
		__syncthreads();
	}
	if (threadId == 0)
		input[index] = localVars[threadId];
	__syncthreads();
}
__global__ void collect_res_gpu_sh(float* input, int numOfBlocks) {
	unsigned int threadId = threadIdx.x;
	unsigned i;
	__shared__ float localVars[1024];
	localVars[threadId] = 0;
	__syncthreads();
	// collect the result of the various blocks
	for (i = 0; i < numOfBlocks; i += blockDim.x) {
		if ((threadId + 1) * blockDim.x < DIM) {
			localVars[threadId] += input[(threadId + i) * blockDim.x];
		}
		__syncthreads();
	}
	// compute the parallel reduction for the collected data 
	for (i = blockDim.x / 2; i > 0; i >>= 1) {
		if (threadId < i) {
			localVars[threadId] += localVars[threadId + i];
		}
		__syncthreads();
	}
	if (threadId == 0) {
		input[threadId] = localVars[threadId];
	}
	__syncthreads();
}
void parallelReductionShared() {
	double start_cpu, end_cpu, start_gpu, end_gpu;
	int BLOCKDIM = 512;
	int dim = DIM_SH;
	printf("please enter the desired block dimension");
	scanf(" %d", &BLOCKDIM);
	srand(time(NULL));

	// allocation data for software results on cpu
	float* p_sum_sw = (float*)malloc(sizeof(float) * dim);
	// allocation data for software input on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);
	float res_hw;
	int i;

	for (i = 0; i < dim; i++) {
		input_v[i] = rand() % 100; // generate random DIM inputs 
	}

	float* d_input;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim); // run the CPU algorithm
	end_cpu = seconds();

	// allocate space for the input array on the GPU
	CHECK(cudaMalloc(&d_input, dim * sizeof(float)));
	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice));

	start_gpu = seconds();

	dim3 blockPerGrid((dim + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);

	// call the reduction for all the block 
	p_sum_gpu_shared << <blockPerGrid, threadsPerBlock >> > (d_input);
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	// finish the results collection using a single block 
	collect_res_gpu_sh << <1, threadsPerBlock >> > (d_input, blockPerGrid.x);
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();
	// retrieve the results from the gpu
	CHECK(cudaMemcpy(&res_hw, d_input, sizeof(float), cudaMemcpyDeviceToHost));

	if (res_hw != p_sum_sw[dim - 1]) {
		// check only for the last variable
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[dim - 1], res_hw);
		return;
	}

	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", dim);

	double cpu_time = end_cpu - start_cpu;
	double gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input)); // free gpu memory

}