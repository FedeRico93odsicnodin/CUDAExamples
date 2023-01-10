#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common.h"

#define DIM_SH 204800;
#define A_DIM 102400
#define B_DIM 204800

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
// optimized parallel reduction computation for each gpu block using multiple threads
__global__ void p_sum_gpu_shared_4(float* input) {
	unsigned threadId = threadIdx.x;
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	// we don't need more than this, since we can have 512 threads at most 
	__shared__ float localVars[512];

	if (threadId < blockDim.x / 2) {
		localVars[threadId] = input[index] + input[index + blockDim.x / 2];
	}
	__syncthreads();

	for (unsigned i = blockDim.x / 4; i > 0; i >>= 1) {
		if (threadId < i) {
			localVars[threadId] += localVars[threadId + i];
		}
		__syncthreads();
	}

	if (threadId == 0)
		input[index] = localVars[threadId];
	__syncthreads();
}
void parallelReductionSharedOptimized() {
	double start_cpu, end_cpu, start_gpu, end_gpu; // cpu and gpu times
	int BLOCKDIM = 512;
	int dim = DIM_SH;
	printf("please enter the desired block dimension");
	scanf(" %d", &BLOCKDIM);
	srand(time(NULL));

	float* p_sum_sw = (float*)malloc(sizeof(float) * dim);      // allocate data for software results on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);       // allocate data for software input on cpu
	float res_hw;
	int i;

	for (i = 0; i < dim; i++)
	{
		input_v[i] = rand() % 100; // generate random DIM inputs
	}

	float* d_input;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim); // run the CPU algorithm
	end_cpu = seconds();

	CHECK(cudaMalloc(&d_input, dim * sizeof(float)));      // allocate space for the input array on the GPU
	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice)); // copy input data on the gpu

	start_gpu = seconds();

	dim3 blocksPerGrid((dim + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);

	p_sum_gpu_shared_4 << <blocksPerGrid, threadsPerBlock >> > (d_input); // call the reduction for all the blocks
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	collect_res_gpu_sh << <1, threadsPerBlock >> > (d_input, blocksPerGrid.x); // finish the results collection using a single block
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();
	CHECK(cudaMemcpy(&res_hw, d_input, sizeof(float), cudaMemcpyDeviceToHost)); // retrieve the results from the GPU

	if (res_hw != p_sum_sw[dim - 1])
	{
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[dim - 1], res_hw); // check only for the last variable
		return;
	}

	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", dim);

	double cpu_time = end_cpu - start_cpu;
	double gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input)); // free gpu memory

}
// compute a parallel redution for each gpu block using multiple threads (optimized warps)
__global__ void p_sum_gpu_optimizedwarp(float* input) {
	unsigned threadId = threadIdx.x;
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	// we don't need more than this, since we can have 512 threads at most 
	__shared__ float localVars[512];
	if (threadId < blockDim.x / 2) {
		localVars[threadId] = input[index] + input[index + blockDim.x / 2];
	}
	__syncthreads();

	for (unsigned i = blockDim.x / 4; i > 32; i >>= 1) {
		if (threadId < i) {
			localVars[threadId] += localVars[threadId + i];
		}
		__syncthreads();
	}
	if (threadId < 32)
		warpReduce(localVars, threadId);
	__syncthreads();

	if (threadId == 0)
		input[index] = localVars[threadId];
	__syncthreads();
}
void parallelReductionSharedOptimizedWarp() {
	double start_cpu, end_cpu, start_gpu, end_gpu; // cpu and gpu times
	int BLOCKDIM = 512;
	int dim = DIM_SH;
	printf("please enter the desired block dimension");
	scanf(" %d", &BLOCKDIM);
	srand(time(NULL));

	float* p_sum_sw = (float*)malloc(sizeof(float) * dim);      // allocate data for software results on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);       // allocate data for software input on cpu
	float res_hw;
	int i;

	for (i = 0; i < dim; i++)
	{
		input_v[i] = rand() % 100; // generate random DIM inputs
	}

	float* d_input;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim); // run the CPU algorithm
	end_cpu = seconds();

	CHECK(cudaMalloc(&d_input, dim * sizeof(float)));      // allocate space for the input array on the GPU
	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice)); // copy input data on the gpu

	start_gpu = seconds();

	dim3 blocksPerGrid((DIM + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);

	p_sum_gpu_optimizedwarp << <blocksPerGrid, threadsPerBlock >> > (d_input); // call the reduction for all the blocks
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	collect_res_gpu_sh << <1, threadsPerBlock >> > (d_input, blocksPerGrid.x); // finish the results collection using a single block
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();

	CHECK(cudaMemcpy(&res_hw, d_input, sizeof(float), cudaMemcpyDeviceToHost)); // retrieve the results from the GPU

	if (res_hw != p_sum_sw[DIM - 1])
	{
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[DIM - 1], res_hw); // check only for the last variable
		return;
	}

	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", DIM);

	double cpu_time = end_cpu - start_cpu;
	double gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input)); // free gpu memory

}
// 4 MERGE SORT IMPLEMENTATION WITH SHARED MEMORY
/*****
Implement a simple merge sort.
The objective of the algorithm is to take two already ordered arrays
and merge them into a single ordered array in the fastest way possible
using also the fact that the two arrays are already ordered within themselves
*****/
__device__ int gpu_ceil2(int a, int b)
{
	return (a + b - 1) / b;
}
// perform a sequential merge sort
__device__ __host__ void merge_sequential2(int* A, int m, int* B, int n, int* C)
{
	int i = 0;
	int j = 0;
	int k = 0;

	while ((i < m) && (j < n))
	{
		if (A[i] <= B[j])
		{
			C[k] = A[i];
			k++, i++;
		}
		else
		{
			C[k] = B[j];
			k++, j++;
		}
	}
	if (i == m)
	{
		for (; j < n; j++)
		{
			C[k] = B[j];
			k++;
		}
	}
	else
	{
		for (; i < m; i++)
		{
			C[k] = A[i];
			k++;
		}
	}
}
// perform the ranking
__device__ int co_rank2(int k, int* A, int m, int* B, int n)
{
	int i = min(k, m);
	int j = k - i;
	int i_low = max(0, k - n);
	int j_low = max(0, k - m);
	int delta;
	bool active = true;
	while (active)
	{
		if (i > 0 && j < n && A[i - 1] > B[j])
		{
			delta = gpu_ceil(i - i_low, 2);
			j_low = j;
			j = j + delta;
			i = i - delta;
		}
		else if (j > 0 && i < m && B[j - 1] >= A[i])
		{
			delta = ((j - j_low + 1) >> 1);
			i_low = i;
			i = i + delta;
			j = j - delta;
		}
		else
		{
			active = false;
		}
	}
	return i;
}
__global__ void merge_basic_kernel2(int* A, int m, int* B, int n, int* C, int tile_size)
{
	/* shared memory allocation */
	extern __shared__ int tile[];
	int* A_T = &tile[0];                                                     // Atile is first half of tile with tile_size elements
	int* B_T = &tile[tile_size];                                             // Btile is second half of tile with tile_size elements   

	int C_curr = blockIdx.x * gpu_ceil2((m + n), gridDim.x);                  // starting point of the C subarray for current block
	int C_next = min((blockIdx.x + 1) * gpu_ceil2((m + n), gridDim.x), (m + n)); // starting point for next block
	if (threadIdx.x == 0)
	{
		tile[0] = co_rank2(C_curr, A, m, B, n); // Make the block-level co-rank values visible to
		tile[1] = co_rank2(C_next, A, m, B, n); // other threads in the block
	}
	__syncthreads();
	int A_curr = tile[0];
	int A_next = tile[1];
	int B_curr = C_curr - A_curr;
	int B_next = C_next - A_next;
	__syncthreads();
	int counter = 0; // iteration counter
	int C_length = C_next - C_curr;
	int A_length = A_next - A_curr;
	int B_length = B_next - B_curr;
	int total_iteration = gpu_ceil2((C_length), tile_size); // total iteration
	int C_completed = 0;
	int A_consumed = 0;
	int B_consumed = 0;
	while (counter < total_iteration)
	{ /* loading tile-size A and B elements into working (tile) memory */
		for (int i = 0; i < tile_size; i += blockDim.x)
		{
			if (i + threadIdx.x < A_length - A_consumed)
			{
				A_T[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
			}
		}
		for (int i = 0; i < tile_size; i += blockDim.x) {
			if (i + threadIdx.x < B_length - B_consumed)
			{
				B_T[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
			}
		}
		__syncthreads();
		int c_curr = threadIdx.x * (tile_size / blockDim.x);
		int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
		c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
		c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed; /* find co-rank for c_curr and c_next */
		int a_curr = co_rank2(c_curr, A_T, min(tile_size, A_length - A_consumed), B_T, min(tile_size, B_length - B_consumed));
		int b_curr = c_curr - a_curr;
		int a_next = co_rank2(c_next, A_T, min(tile_size, A_length - A_consumed), B_T, min(tile_size, B_length - B_consumed));
		int b_next = c_next - a_next;                                                                                      /* All threads call the sequential merge function */
		merge_sequential2(A_T + a_curr, a_next - a_curr, B_T + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr); /* Update the A and B elements that have been consumed thus far */
		counter++;
		C_completed += tile_size;
		A_consumed += co_rank2(tile_size, A_T, tile_size, B_T, tile_size);
		B_consumed = C_completed - A_consumed;
		__syncthreads();
	}
}
// simple sort, just to sort the randomly generated arrays
void sort_array2(int* arr, int dim)
{
	int i, j;

	for (i = 0; i < dim; i++)
	{
		for (j = i + 1; j < dim; j++)
		{
			if (arr[j] < arr[i])
			{
				int tmp = arr[i];
				arr[i] = arr[j];
				arr[j] = tmp;
			}
		}
	}
}
void mergeSortTiledShared() {
	double start_cpu, end_cpu, start_gpu, end_gpu;

	int i;
	int BLOCKSIZE = 512;
	int TILE_S = 5;
	printf("please enter the desired block dimension");
	scanf(" %d", &BLOCKSIZE);
	printf("please enter the desired the tile S");
	scanf(" %d", &TILE_S);

	int* A = (int*)malloc(sizeof(int) * A_DIM);
	int* B = (int*)malloc(sizeof(int) * B_DIM);
	int* output = (int*)malloc(sizeof(int) * (A_DIM + B_DIM));

	// allocate GPU memory
	int* A_d, * B_d, * output_d;
	CHECK(cudaMalloc(&A_d, sizeof(int) * A_DIM));
	CHECK(cudaMalloc(&B_d, sizeof(int) * B_DIM));
	CHECK(cudaMalloc(&output_d, sizeof(int) * (A_DIM + B_DIM)));

	// gpu results on sw
	int* output_gpu = (int*)malloc(sizeof(int) * (A_DIM + B_DIM));

	// generate randomly 2 arrays and sort them (input for the merge sort are 2 already sorted arrays)
	srand(0);
	for (i = 0; i < A_DIM; i++)
	{
		A[i] = rand() % 50;
	}
	sort_array(A, A_DIM);
	for (i = 0; i < B_DIM; i++)
	{
		B[i] = rand() % 50;
	}
	sort_array(B, B_DIM);

	// sort them on CPU
	start_cpu = seconds();
	merge_sequential2(A, A_DIM, B, B_DIM, output);
	end_cpu = seconds();

	// copy inputs to GPU
	CHECK(cudaMemcpy(A_d, A, A_DIM * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(B_d, B, B_DIM * sizeof(int), cudaMemcpyHostToDevice));

	dim3 blocksPerGrid((A_DIM + B_DIM + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);
	dim3 threadsPerBlock(BLOCKSIZE, 1, 1);

	start_gpu = seconds();
	merge_basic_kernel2 << <blocksPerGrid, threadsPerBlock, TILE_S * sizeof(int) >> > (A_d, A_DIM, B_d, B_DIM, output_d, TILE_S);

	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();
	CHECK(cudaMemcpy(output_gpu, output_d, (A_DIM + B_DIM) * sizeof(int), cudaMemcpyDeviceToHost)); // copy back the results

	// check the sorted array
	for (i = 0; i < A_DIM + B_DIM; i++)
	{
		if (output_gpu[i] != output[i])
		{
			printf("Error in position: %d\n", i);
			break;
		}
	}

	// print sort time
	printf("Sort Time CPU: %.10lf\n", end_cpu - start_cpu);
	printf("Sort Time GPU: %.10lf\n", end_gpu - start_gpu);
}