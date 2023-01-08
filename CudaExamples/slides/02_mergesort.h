#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common.h"

#define MAXVAL 50
#define A_DIM_2 102400
#define B_DIM_2 204800
// IMPLEMENTATION SINGLE THREAD
/*****
Implement a simple merge sort.
The objective of the algorithm is to take two already ordered arrays
and merge them into a single ordered array in the fastest way possible
using also the fact that the two arrays are already ordered within themselves
*****/
__device__ int gpu_ceil(int a, int b) {
	return (a + b - 1) / b;
}
// perform a sequential merge sort 
__host__ __device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
	int i = 0;
	int j = 0;
	int k = 0;

	while ((i < m) && (j < n)) {
		if (A[i] <= B[j]) {
			C[k] = A[i];
			k++, i++;
		}
		else {
			C[k] = B[j];
			k++, j++;
		}
	}
	if (i == m) {
		for (; j < n; j++) {
			C[k] = B[j];
			k++;
		}
	}
	else {
		for (; i < m; i++) {
			C[k] = A[i];
			k++;
		}
	}
}
__global__ void merge_sequential_gpu(int* A, int m, int* B, int n, int* C) {
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
// simple sort, just to sort the randomly generated arrays
void sort_array(int* arr, int dim)
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
void sequentialMergeSort() {
    double start_cpu, end_cpu, start_gpu, end_gpu;

    int i;
    int A_DIM;
    int B_DIM;
    // taking the dimensions for the 2 arrays 
    printf("give the A dimension:");
    scanf(" %d", &A_DIM);
    printf("give the B dimension:");
    scanf(" %d", &B_DIM);

    // allocation for the CPU
    int* A = (int*)malloc(sizeof(int) * A_DIM);
    int* B = (int*)malloc(sizeof(int) * B_DIM);
    int* output = (int*)malloc(sizeof(int) * (A_DIM + B_DIM));

    // allocation for the GPU 
    int* A_d, * B_d, * output_d;
    CHECK(cudaMalloc(&A_d, sizeof(int) * A_DIM));
    CHECK(cudaMalloc(&B_d, sizeof(int) * B_DIM));
    CHECK(cudaMalloc(&output_d, sizeof(int) * (A_DIM + B_DIM)));

    // gpu results on sw 
    int* output_gpu = (int*)malloc(sizeof(int) * (A_DIM + B_DIM));

    // generate randomly 2 arrays and sort them (input for the merge sort are 2 already sorted arrays)
    srand(0);
    for (i = 0; i < A_DIM; i++) {
        A[i] = rand() % MAXVAL;
    }
    sort_array(A, A_DIM);
    for (i = 0; i < B_DIM; i++) {
        B[i] = rand() % MAXVAL;
    }
    sort_array(B, B_DIM);

    // print the 2 ordered input arrays 
    printf("Input array A: [");
    for (i = 0; i < A_DIM; i++) {
        if (i != A_DIM - 1)
            printf("%d,", A[i]);
        else
            printf("%d", A[i]);
    }
    printf("]\n");

    printf("Input Array B: [");
    for (i = 0; i < B_DIM; i++) {
        if (i != B_DIM - 1)
            printf("%d,", B[i]);
        else
            printf("%d", B[i]);
    }
    printf("]\n");

    // sort them on CPU
    start_cpu = seconds();
    merge_sequential(A, A_DIM, B, B_DIM, output);
    end_cpu = seconds();

    // copy inputs to GPU
    CHECK(cudaMemcpy(A_d, A, A_DIM * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B, B_DIM * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(1, 1, 1);

    start_gpu = seconds();
    merge_sequential_gpu << <blocksPerGrid, threadsPerBlock >> > (A_d, A_DIM, B_d, B_DIM, output_d);

    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    end_gpu = seconds();

    CHECK(cudaMemcpy(output_gpu, output_d, (A_DIM + B_DIM) * sizeof(int), cudaMemcpyDeviceToHost));

    // print the sorted merged array
    for (i = 0; i < A_DIM + B_DIM; i++) {
        if (output_gpu[i] != output[i]) {
            printf("Error in position: %d\n", i);
            break;
        }

        if (i != A_DIM + B_DIM - 1)
            printf("%d,", output_gpu[i]);
        else 
            printf("%d", output_gpu[i]);

    }
    printf("]\n");
    // print sort time 
    printf("Sort time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("Sort time GPU: %.10lf\n", end_gpu - start_gpu);
}
// 02 MERGE SORT RANK SINGLE BLOCK 
/*****
Implement a simple merge sort.
The objective of the algorithm is to take two already ordered arrays
and merge them into a single ordered array in the fastest way possible
using also the fact that the two arrays are already ordered within themselves
*****/
__device__ int co_rank(int k, int* A, int m, int* B, int n) {
    int i = min(k, m);
    int j = k - i;
    int i_low = max(0, k - n);
    int j_low = max(0, k - m);
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = gpu_ceil(i - i_low, 2);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else {
            active = false;
        }
        return i;
    }
}
__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = threadIdx.x;
    int k_curr = tid * gpu_ceil(m + n, blockDim.x);
    int k_next = min((tid + 1) * gpu_ceil(m + n, blockDim.x), m + n);

    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    // all threads call the sequential merge function 
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}
void mergeSortRankBlock() {
    double start_cpu, end_cpu, start_gpu, end_gpu;
    int i;
    int blockDim = 512;
    printf("digit the dimension of the block:");
    scanf("%d", &blockDim);
    int BLOCKSIZE = blockDim;

    int* A = (int*)malloc(sizeof(int) * A_DIM_2);
    int* B = (int*)malloc(sizeof(int) * B_DIM_2);
    int* output = (int*)malloc(sizeof(int) * (A_DIM_2 + B_DIM_2));

    // allocate GPU memory

    int* A_d, * B_d, * output_d;
    CHECK(cudaMalloc(&A_d, sizeof(int) * A_DIM_2));
    CHECK(cudaMalloc(&B_d, sizeof(int) * B_DIM_2));
    CHECK(cudaMalloc(&output_d, sizeof(int) * (A_DIM_2 + B_DIM_2)));
    // gpu results on sw
    int* output_gpu = (int*)malloc(sizeof(int) * (A_DIM_2 + B_DIM_2));

    // generate randomly 2 arrays and sort them (input for the merge sort are 2 already sorted arrays)
    srand(0);
    for (i = 0; i < A_DIM_2; i++)
    {
        A[i] = rand() % MAXVAL;
    }
    sort_array(A, A_DIM_2);
    for (i = 0; i < B_DIM_2; i++)
    {
        B[i] = rand() % MAXVAL;
    }
    sort_array(B, B_DIM_2);
    // sort them on CPU
    start_cpu = seconds();
    merge_sequential(A, A_DIM_2, B, B_DIM_2, output);
    end_cpu = seconds();

    // copy inputs to GPU
    CHECK(cudaMemcpy(A_d, A, A_DIM_2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B, B_DIM_2 * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(BLOCKSIZE, 1, 1);

    start_gpu = seconds();
    merge_basic_kernel << <blocksPerGrid, threadsPerBlock >> > (A_d, A_DIM_2, B_d, B_DIM_2, output_d);

    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    end_gpu = seconds();

    CHECK(cudaMemcpy(output_gpu, output_d, (A_DIM_2 + B_DIM_2) * sizeof(int), cudaMemcpyDeviceToHost)); // copy back the results

    // check the sorted array
    for (i = 0; i < A_DIM_2 + B_DIM_2; i++)
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