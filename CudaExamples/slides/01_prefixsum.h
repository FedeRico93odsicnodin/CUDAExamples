#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common.h"

#define DIMTEST 20
#define DIM 204800

/*first version of the prefix sum using a single thread*/
void p_sum_cpu(float* p_sum, float* input, int length, bool consoleEnabled) {
	p_sum[0] = input[0];
	int i;
	for (i = 1; i < length; ++i) {
		if(consoleEnabled)
			printf("p_sum[%d]: %f - input[%d]: %f\n", i - 1, p_sum[i - 1], i, input[i]);
		p_sum[i] = p_sum[i - 1] + input[i];
	}
}
__global__ void p_sum_gpu(float* p_sum, float* input, int length, bool consoleEnabled) {
	p_sum[0] = input[0];
	int i;
	for (i = 1; i < length; ++i) {
		if (consoleEnabled)
			printf("p_sum[%d]: %f - input[%d]: %f\n", i - 1, p_sum[i - 1], i, input[i]);
		p_sum[i] = p_sum[i - 1] + input[i];
	}
}
void prefixSumFirstVersion() {
	
	int tmp = 0;
	printf("enable console mode?");
	scanf("%d", &tmp);
	bool en = tmp;

	double start_cpu, end_cpu, start_gpu, end_gpu;
	int dim = DIM;
	if (en)
		dim = DIMTEST;
	srand(time(NULL));

	float* p_sum_sw = (float*)malloc(sizeof(float) * dim);
	float* input_v = (float*)malloc(sizeof(float) * dim);
	float* p_sum_hw = (float*)malloc(sizeof(float) * dim);
	float* reset = (float*)malloc(sizeof(float) * dim);
	int i;

	// initializing input 
	for (i = 0; i < dim; i++) {
		input_v[i] = rand() % 100;
		reset[i] = 0.0f;
		p_sum_sw[i] = 0.0f;
		if(en)
			printf("input[%d] = %f, reset[%d] = %f, p_sum_sw[%d] = %f\n"
				, i, input_v[i], i, reset[i], i, p_sum_sw[i]);
	}

	// monitoring time for the CPU
	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim, en);
	end_cpu = seconds();

	// allocating GPU memory 
	float* d_input, * d_p_sum;
	CHECK(cudaMalloc(&d_input, dim * sizeof(float)));
	CHECK(cudaMalloc(&d_p_sum, dim * sizeof(float)));

	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_p_sum, reset, dim * sizeof(float), cudaMemcpyHostToDevice));

	start_gpu = seconds();
	dim3 blockPerGrid(1, 1, 1);
	dim3 threadsPerBlock(1, 1, 1);
	p_sum_gpu << <blockPerGrid, threadsPerBlock >> > (d_p_sum, d_input, dim, en);
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();

	CHECK(cudaMemcpy(p_sum_hw, d_p_sum, dim * sizeof(float), cudaMemcpyDeviceToHost));
	if (p_sum_hw[DIM - 1] != p_sum_sw[DIM - 1]) {
		// check only for the last variable 
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[DIM - 1], p_sum_hw[DIM - 1]);
		return;
	}

	printf("ALL RESULT CORRECT, VEC DIM = %d\n", DIM);

	double gpu_time = 0, cpu_time = 0;

	cpu_time = end_cpu - start_cpu;
	gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input));
	CHECK(cudaFree(d_p_sum));
}
/*second version of the prefix sum using a single block where each thread i computes
the sum between the first element (position 0) and the one with position i*/
__global__ void p_sum_gpu_2(float* p_sum, float* input, int segLength, int length) {
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

	int stride;
	for (stride = 0; stride < length; stride++) {
		// each thread compute its own p_sum cell, reading if needed a variable from the original input 
		if (index >= stride)
			p_sum[index] += input[index - stride];
	}
}
void prefixSumSecondVersion() {
	
	int blockDim = 512;
	printf("digit the dimension of the block:");
	scanf("%d", &blockDim);

	int tmp = 0;
	printf("enable console mode?");
	scanf("%d", &tmp);
	bool en = tmp;

	double start_cpu, end_cpu, start_gpu, end_gpu; // cpu and gpu times 
	int BLOCKDIM = blockDim;
	int dim = DIM;
	if (en)
		dim = DIMTEST;
	srand(time(NULL));

	// allocate data for software results on cpu
	float* p_sum_sw = (float*)malloc(sizeof(float) * dim);
	// allocate data for software input on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);
	// allocate data for hardware results on cpu
	float* p_sum_hw = (float*) malloc(sizeof(float) * dim);

	int i;
	for (i = 0; i < dim; i++) {
		input_v[i] = rand() % 100; // generate random DIM inputs 
		//p_sum_sw[i] = 0.0f;
	}

	float* d_input, * d_p_sum;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim, en); // run the CPU algorithm
	end_cpu = seconds();
	
	// cuda initializations 
	CHECK(cudaMalloc(&d_input, dim * sizeof(float)));
	CHECK(cudaMalloc(&d_p_sum, dim * sizeof(float)));
	// send input data to the gpu
	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice));
	// set partial sum results to zero before executing the kernel
	CHECK(cudaMemset(d_p_sum, 0, dim));

	start_gpu = seconds();
	// setting a number of blocks to equally distribute the load
	dim3 blockPerGrid((DIM + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	// num of threads is equal to the specified input 
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);
	// run the gpu kernel 
	p_sum_gpu_2 << <blockPerGrid, threadsPerBlock >> > (d_p_sum, d_input, dim / BLOCKDIM, dim);
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize()); // sync to take kernel time 
	end_gpu = seconds();

	CHECK(cudaMemcpy(p_sum_hw, d_p_sum, dim * sizeof(float), cudaMemcpyDeviceToHost));

	if (p_sum_hw[DIM - 1] != p_sum_sw[DIM - 1]) {
		// check only for the last variable 
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[DIM - 1], p_sum_hw[DIM - 1]);
	}

	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", DIM);

	double gpu_time = 0, cpu_time = 0;
	cpu_time = end_cpu - start_cpu;
	gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	// free gpu memory 
	CHECK(cudaFree(d_input));
	CHECK(cudaFree(d_p_sum));

}
/*third version of the prefix sum; this version makes a first attempt of using a
parallel reduction. DO NOTE: this code works only with a single block (even if it asks
for the block size and coherently computes the grid size*/
/*
* at iteration 1:
* thread 1 sums input[0] to input[1]
* thread 3 sums input[2] to input[3]
* thread 5 sums input[4] to input[5]
* thread 7 sums input[8] to input[7]
* ...
* at iteration 2:
* thread 3 sums input[1] to input[3]
* thread 7 sums input[5] to input[7]
* ...
* at iteration 3:
* thread 7 sums input[3] to input[7]
* ...
*/

/*
* considering a block with 64 threads
* at iteration 1:
* thread 1 sums input[0] to input[1]
* thread 3 sums input[2] to input[3]
* thread 5 sums input[4] to input[5]
* thread 7 sums input[8] to input[7]
* ...
* thread 31 sums input[30] to input[31]
* all other threads do nothing
*
* at iteration 2:
* thread 3 sums input[1] to input[3]
* thread 7 sums input[5] to input[7]
* ...
* thread 31 sums input[29] to input[31]
* all other threads do nothing
*
* at iteration 3:
* thread 7 sums input[3] to input[7]
* thread 15 sums input[11] to input[15]
* ...
* thread 31 sums input[27] to input[31]
* all other threads do nothing
* ...
* This may kernel may be optimized by defining a grid with a
* size equal to a half of the data size
*/
__global__ void p_sum_gpu_3(float* input, bool enableConsole) {
	unsigned int threadId = threadIdx.x;
	// TODO: reporting this code in a debug 
	for (unsigned i = 1; i < blockDim.x; i <<= 1) {
		if (((threadId % (i * 2)) == (i * 2) - 1)) {
			if(enableConsole)
				printf("i = %d, threadId = %d\n", i, threadId);
			input[threadId] += input[threadId - 1];
		}
		__syncthreads();
	}
}
void prefixSumThirdVersion() {
	double start_cpu, end_cpu, start_gpu, end_gpu; // cpu and gpu times

	int blockDim = 512;
	printf("digit the dimension of the block:");
	scanf("%d", &blockDim);
	int BLOCKDIM = blockDim;
	int tmp = 0;
	printf("enable console mode?");
	scanf("%d", &tmp);
	bool en = tmp;

	int dim = DIM;
	if (en)
		dim = DIMTEST;
	srand(time(NULL));

	float* p_sum_sw = (float*)malloc(sizeof(float) * dim); // allocate data for software results on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);  // allocate data for software input on cpu
	float* p_sum_hw = (float*)malloc(sizeof(float) * dim); // allocate data for hardware results on cpu

	int i;
	for (i = 0; i < dim; i++)
	{
		input_v[i] = rand() % 100; // generate random DIM inputs
	}
	float* d_input;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim, en); // run the CPU algorithm
	end_cpu = seconds();

	CHECK(cudaMalloc(&d_input, dim * sizeof(float))); // allocate space only for an array on the GPU
	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice)); // copy input data on the gpu

	start_gpu = seconds();

	dim3 blocksPerGrid((DIM + BLOCKDIM - 1) / BLOCKDIM, 1, 1); // block number is the same as before
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);
	p_sum_gpu_3 << <blocksPerGrid, threadsPerBlock >> > (d_input, en);
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();

	CHECK(cudaMemcpy(p_sum_hw, d_input, dim * sizeof(float), cudaMemcpyDeviceToHost)); // copy back the results

	if (p_sum_hw[DIM - 1] != p_sum_sw[DIM - 1])
	{
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[DIM - 1], p_sum_hw[DIM - 1]); // check only for the last variable
		return;
	}
	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", DIM);

	double cpu_time = end_cpu - start_cpu;
	double gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input)); // free gpu memory

}
/*forth version of the prefix sum; this version is a better implementation of the
parallel reduction. DO NOTE: this code works only with a single block (even if it asks
for the block size and coherently computes the grid size*/
/*
* considering a block with 64 threads
* at iteration 1:
* thread 0 sums input[0] to input[32+0]
* thread 1 sums input[1] to input[32+1]
* thread 2 sums input[2] to input[32+2]
* thread 3 sums input[3] to input[32+3]
* ...
* thread 31 sums input[31] to input[32+31]
* all other threads do nothing
*
* at iteration 2:
* thread 0 sums input[0] to input[16+0]
* thread 1 sums input[1] to input[16+1]
* thread 2 sums input[2] to input[16+2]
* thread 3 sums input[3] to input[16+3]
* ...
* thread 15 sums input[15] to input[16+15]
* all other threads do nothing
*
* at iteration 3:
* thread 0 sums input[0] to input[8+0]
* thread 1 sums input[1] to input[8+1]
* thread 2 sums input[2] to input[8+2]
* thread 3 sums input[3] to input[8+3]
* ...
* thread 7 sums input[15] to input[8+7]
* all other threads do nothing
* ...
* This may kernel may be optimized by defining a grid with a
* size equal to a half of the data size
*/
__global__ void p_sum_gpu_4(float* input, bool enableConsole) { // compute a parallel reduction for each gpu block using multiple threads
	unsigned threadId = threadIdx.x;
	// TODO: reporting this code in a debug 
	for (unsigned i = blockDim.x / 2; i > 0; i >>= 1) {
		if (threadId < i) {
			if(enableConsole)
				printf("%d threadId: SUM of input[%d] = %f with input[%d] = %f\n"
				, threadId, threadId, input[threadId], threadId + 1, input[threadId + 1]);
			input[threadId] += input[threadId + i];
		}
		__syncthreads();
	}
}
void prefixSumFourthVersion() {
	double start_cpu, end_cpu, start_gpu, end_gpu; // cpu and gpu times

	int blockDim = 512;
	printf("digit the dimension of the block:");
	scanf("%d", &blockDim);
	int BLOCKDIM = blockDim;
	int tmp = 0;
	printf("enable console mode?");
	scanf("%d", &tmp);
	bool en = tmp;

	int dim = DIM;
	if (en)
		dim = DIMTEST;
	srand(time(NULL));

	float* p_sum_sw = (float*)malloc(sizeof(float) * dim); // allocate data for software results on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);  // allocate data for software input on cpu
	float* p_sum_hw = (float*)malloc(sizeof(float) * dim); // allocate data for hardware results on cpu

	int i;

	for (i = 0; i < dim; i++)
	{
		input_v[i] = rand() % 100; // generate random DIM inputs
	}
	float* d_input;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim, en); // run the CPU algorithm
	end_cpu = seconds();

	CHECK(cudaMalloc(&d_input, dim * sizeof(float))); // allocate space only for an array on the GPU

	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice)); // copy input data on the gpu


	start_gpu = seconds();

	dim3 blocksPerGrid((DIM + BLOCKDIM - 1) / BLOCKDIM, 1, 1); // block number is the same as before
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);

	p_sum_gpu_4 << <blocksPerGrid, threadsPerBlock >> > (d_input, en);
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();
	CHECK(cudaMemcpy(p_sum_hw, d_input, dim * sizeof(float), cudaMemcpyDeviceToHost)); // copy back the results

	if (p_sum_hw[DIM - 1] != p_sum_sw[DIM - 1])
	{
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[DIM - 1], p_sum_hw[0]); // check only for the last variable
		return;
	}
	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", DIM);

	double cpu_time = end_cpu - start_cpu;
	double gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input)); // free gpu memory

}

/*fifth version of the prefix sum supports the execution of the kernel with a grid
with multiple blocks. This version uses the parallel reduction at block level
then it uses a second kernel to aggregate the results produced by the various blocks;
this second kernel performs a single-block parallel reduction as well.*/
__global__ void p_sum_gpu_5(float* input) { // compute a parallel reduction for each gpu block using multiple threads
	unsigned threadId = threadIdx.x;
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	// TODO: eventuale simulazione in debug 
	for (unsigned i = blockDim.x / 2; i > 0; i >>= 1) {
		if (threadId < i) {
			input[index] += input[index + i];
		}
		__syncthreads();
	}
}
__global__ void collect_res_gpu(float* input, float* p_sum, int numOfBlocks) {
	// compute the final reduction
	unsigned int threadId = threadIdx.x;
	unsigned i;
	for (i = 0; i < numOfBlocks; i += blockDim.x) // collect the result of the various blocks 
	{
		if ((threadId + i) * blockDim.x < DIM) {
			p_sum[threadId] += input[(threadId + i) * blockDim.x];
		}
		__syncthreads();
	}

	for (i = blockDim.x / 2; i > 0; i >>= 1) {
		if (threadId < i) {
			p_sum[threadId] += p_sum[threadId + i];
		}
		__syncthreads();
	}
}
void prefixSumFifthVersion() {
	double start_cpu, end_cpu, start_gpu, end_gpu; // cpu and gpu times

	int blockDim = 512;
	printf("digit the dimension of the block:");
	scanf("%d", &blockDim);
	int BLOCKDIM = blockDim;
	int tmp = 0;
	printf("enable console mode?");
	scanf("%d", &tmp);
	bool en = tmp;

	int dim = DIM;
	if (en)
		dim = DIMTEST;
	srand(time(NULL));

	float* p_sum_sw = (float*)malloc(sizeof(float) * dim);      // allocate data for software results on cpu
	float* input_v = (float*)malloc(sizeof(float) * dim);       // allocate data for software input on cpu
	float* p_sum_hw = (float*)malloc(sizeof(float) * BLOCKDIM); // allocate data for hardware results on cpu
	int i;

	for (i = 0; i < dim; i++)
	{
		input_v[i] = rand() % 100; // generate random DIM inputs
	}

	float* d_input, * d_p_sum;

	start_cpu = seconds();
	p_sum_cpu(p_sum_sw, input_v, dim, en); // run the CPU algorithm
	end_cpu = seconds();

	CHECK(cudaMalloc(&d_input, dim * sizeof(float)));      // allocate space for the input array on the GPU
	CHECK(cudaMalloc(&d_p_sum, BLOCKDIM * sizeof(float))); // allocate space for the result array on the GPU
	CHECK(cudaMemset(d_p_sum, 0, BLOCKDIM));
	CHECK(cudaMemcpy(d_input, input_v, dim * sizeof(float), cudaMemcpyHostToDevice)); // copy input data on the gpu

	start_gpu = seconds();

	dim3 blocksPerGrid((DIM + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
	dim3 threadsPerBlock(BLOCKDIM, 1, 1);

	p_sum_gpu_5 << <blocksPerGrid, threadsPerBlock >> > (d_input); // call the reduction for all the blocks
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	collect_res_gpu << <1, threadsPerBlock >> > (d_input, d_p_sum, blocksPerGrid.x); // finish the results collection using a single block
	CHECK_KERNELCALL();
	CHECK(cudaDeviceSynchronize());
	end_gpu = seconds();
	CHECK(cudaMemcpy(p_sum_hw, d_p_sum, BLOCKDIM * sizeof(float), cudaMemcpyDeviceToHost)); // retrieve the results from the GPU

	if (p_sum_hw[0] != p_sum_sw[DIM - 1])
	{
		printf("WRONG RES ON GPU SW: %f HW: %f \n", p_sum_sw[DIM - 1], p_sum_hw[0]); // check only for the last variable
	}

	printf("ALL RESULTS CORRECT, VEC DIM = %d\n", DIM);

	double cpu_time = end_cpu - start_cpu;
	double gpu_time = end_gpu - start_gpu;
	printf("GPU TIME: %lf\n", gpu_time);
	printf("CPU TIME: %lf\n", cpu_time);

	CHECK(cudaFree(d_input)); // free gpu memory
}