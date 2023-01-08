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
void prefixSumFirstVersion(bool en) {
	
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