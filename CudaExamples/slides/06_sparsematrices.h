#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "../common.h"

void read_matrix(
	int** row_ptr,
	int** col_ind,
	float** values,
	const char* filename,
	int* num_rows,
	int* num_cols,
	int* num_vals);
void read_matrix_v2(
	int** row_ptr,
	int** col_ind,
	float** values,
	const char* filename,
	int* num_rows,
	int* num_cols,
	int* num_vals,
	int** row_ids,
	int** col_ids
);
// parallel SpMV using CSR format 
__global__ void spmv_csr(
	const int* row_ptr,
	const int* col_ind,
	const float* values,
	const int num_rows,
	float* x,
	float* y) {
	// uses a grid-stride loop to perform dot product 
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x * gridDim.x) {
		float dotProduct = 0;
		const int row_start = row_ptr[i];
		const int row_end = row_ptr[i + 1];

		for (int j = row_start; j < row_end; j++) {
			dotProduct += values[j] * x[col_ind[j]];
		}

		y[i] = dotProduct;
	}
}
// parallel SpMV sw implementation 
void spmv_csr_sw(
	const int* row_ptr,
	const int* col_ind,
	const float* values,
	const int num_rows,
	const float* x,
	float* y) {
	for (int i = 0; i < num_rows; i++) {
		float dotProduct = 0;
		const int row_start = row_ptr[i];
		const int row_end = row_ptr[i + 1];

		for (int j = row_start; j < row_end; j++) {
			dotProduct += values[j] * x[col_ind[j]];
		}

		y[i] = dotProduct;
	}
}
// launching the functions 
void spmv_csr_main() {
	int* row_ptr, * col_ind, num_rows, num_cols, num_vals;
	float* values;

	char finput[256];
	printf("please insert for the file location:");
	scanf(" %[^\n]%*c", &finput);

	int BLOCKSIZE = 512;
	printf("please insert the block size value:");
	scanf(" %d", &BLOCKSIZE);


	double start_cpu, end_cpu, start_gpu, end_gpu;

	read_matrix(&row_ptr, &col_ind, &values, finput, &num_rows, &num_cols, &num_vals);

	float* x = (float*)malloc(num_rows * sizeof(float));
	float* y_sw = (float*)malloc(num_rows * sizeof(float));
	float* y_hw = (float*)malloc(num_rows * sizeof(float));

	for (int i = 0; i < num_rows; i++) {
		x[i] = 1.0;
	}

	// compute in sw 
	start_cpu = seconds();
	spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
	end_cpu = seconds();

	// allocate on device 
	int* d_row_ptr, * d_col_ind;
	float* d_values, * d_x, * d_y;
	CHECK(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&d_col_ind, num_vals * sizeof(int)));
	CHECK(cudaMalloc((void**)&d_values, num_vals * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

	// copy from host to device 
	CHECK(cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));
	int currBlockX = 1;
	if (((num_rows - 1) / BLOCKSIZE) != 0)
		currBlockX = (num_rows - 1) / BLOCKSIZE;
	dim3 blocksPerGrid(currBlockX, 1, 1);
	dim3 threadsPerBlock(BLOCKSIZE, 1, 1);
	// calling the kernel function 
	start_gpu = seconds();
	spmv_csr << <blocksPerGrid, threadsPerBlock >> > (d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y);
	end_gpu = seconds();
	CHECK_KERNELCALL();

	CHECK(cudaMemcpy(y_hw, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

	// check results 
	for (int i = 0; i < num_rows; i++) {
		if (y_hw[i] != y_sw[i]) {
			printf("ERROR AT INDEX %d", i);
			break;
		}
	}

	// print time 
	printf("SPMV Time CPU: %.10lf\n", end_cpu - start_cpu);
	printf("SPMV Time GPU: %.10lf\n", end_gpu - start_gpu);

	// Free
	CHECK(cudaFree(d_row_ptr));
	CHECK(cudaFree(d_col_ind));
	CHECK(cudaFree(d_values));
	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));

	free(row_ptr);
	free(col_ind);
	free(values);
	free(y_sw);
	free(y_hw);
}
/// <summary>
/// Reads a sparse matrix and represents it using CSR
/// (Compressed Sparse Row) format 
/// </summary>
/// <param name="row_ptr"></param>
/// <param name="col_ind"></param>
/// <param name="values"></param>
/// <param name="filename"></param>
/// <param name="num_rows"></param>
/// <param name="num_cols"></param>
/// <param name="num_vals"></param>
void read_matrix(
	int** row_ptr,
	int** col_ind,
	float** values,
	const char* filename,
	int* num_rows,
	int* num_cols,
	int* num_vals) 
{
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stdout, "File cannot be opened!\n");
		return;
	}

	// get the number of rows, columns and non zero values 
	fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);

	int* row_ptr_t = (int*)malloc((*num_rows + 1) * sizeof(int));
	int* col_ind_t = (int*)malloc(*num_vals * sizeof(int));
	float* values_t = (float*)malloc(*num_vals * sizeof(float));

	// collect occurrances of each row for determining the indices of row_ptr
	int* row_occurances = (int*)malloc(*num_rows * sizeof(int));
	for (int i = 0; i < *num_rows; i++) {
		row_occurances[i] = 0;
	}

	int row, column;
	float value;
	while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
		row--;
		column--;
		row_occurances[row]++;
	}

	// set row_ptr
	int index = 0;
	for (int i = 0; i < *num_rows; i++) {
		row_ptr_t[i] = index;
		index += row_occurances[i];
	}

	// set the file position to the beginning of the file 
	rewind(file);

	// read the file again, save column indices and values 
	for (int i = 0; i < *num_vals; i++) {
		col_ind_t[i] = -1;
	}

	fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);
	int i = 0;
	while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
		row--;
		column--;

		// Find the correct index (i + row_ptr_t[row]) using both row information and an index i
		while (col_ind_t[i + row_ptr_t[row]] != -1) {
			i++;
		}
		col_ind_t[i + row_ptr_t[row]] = column;
		values_t[i + row_ptr_t[row]] = value;
		i = 0;
	}

	fclose(file);

	*row_ptr = row_ptr_t;
	*col_ind = col_ind_t;
	*values = values_t;
}
// coo implementation of the kernel 
__global__ void spmv_coo(
	const int num_vals,
	const int* row_ids,
	const int* col_ids,
	const float* values,
	const int num_rows,
	const float* x,
	float* y
) {
	// uses a grid-stride loop to perform dot product 
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x * gridDim.x) {
		if (i < num_vals)
			atomicAdd(y + row_ids[i], values[i] * x[col_ids[i]]);
	}
}
// version of main with coo
void spmv_csr_main_coo() {
	int* row_ptr, * col_ind, * row_ids, * col_ids, num_rows, num_cols, num_vals;
	float* values;
	int BLOCKSIZE = 512;
	printf("please insert the block size:");
	scanf(" %d", &BLOCKSIZE);
	char filename[256];
	printf("please insert for the file location:");
	scanf(" %[^\n]%*c", &filename);

	double start_cpu, end_cpu, start_gpu, end_gpu;

	read_matrix_v2(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals, &row_ids, &col_ids);

	float* x = (float*)malloc(num_rows * sizeof(float));
	float* y_sw = (float*)malloc(num_rows * sizeof(float));
	float* y_hw = (float*)malloc(num_rows * sizeof(float));

	for (int i = 0; i < num_rows; i++) {
		x[i] = 1.0;
	}

	// compute in sw 
	start_cpu = seconds();
	spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
	end_cpu = seconds();

	// allocate on device 
	int* d_row_ids, * d_col_ids;
	float* d_values, * d_x, * d_y;
	CHECK(cudaMalloc((void**)&d_row_ids, num_vals * sizeof(int)));
	CHECK(cudaMalloc((void**)&d_col_ids, num_vals * sizeof(int)));
	CHECK(cudaMalloc((void**)&d_values, num_vals * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

	// copy from host to device 
	CHECK(cudaMemcpy(d_row_ids, row_ids, num_vals * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_col_ids, col_ids, num_vals * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocksPerGrid((num_vals - 1) / BLOCKSIZE, 1, 1);
	dim3 threadsPerBlock(BLOCKSIZE, 1, 1);
	// call kernel function 
	start_gpu = seconds();
	spmv_coo << <blocksPerGrid, threadsPerBlock >> > (num_vals, d_row_ids, d_col_ids, d_values, num_rows, d_x, d_y);
	end_gpu = seconds();
	CHECK_KERNELCALL();

	CHECK(cudaMemcpy(y_hw, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

	// print time 
	printf("SPMV Time CPU: %.10lf\n", end_cpu - start_cpu);
	printf("SPMV Time GPU: %.10lf\n", end_gpu - start_gpu);

	// free
	CHECK(cudaFree(d_row_ids));
	CHECK(cudaFree(d_col_ids));
	CHECK(cudaFree(d_values));
	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));

	free(row_ptr);
	free(col_ind);
	free(values);
	free(y_sw);
	free(y_hw);
}
/// <summary>
/// Reads a sparse matrix and represents it using CSR
/// (Compressed Sparse row) format 
/// </summary>
/// <param name="row_ptr"></param>
/// <param name="col_ind"></param>
/// <param name="values"></param>
/// <param name="filename"></param>
/// <param name="num_rows"></param>
/// <param name="num_cols"></param>
/// <param name="num_vals"></param>
/// <param name="row_ids"></param>
/// <param name="col_ids"></param>
void read_matrix_v2(
	int** row_ptr,
	int** col_ind,
	float** values,
	const char* filename,
	int* num_rows,
	int* num_cols,
	int* num_vals,
	int** row_ids,
	int** col_ids
) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stdout, "File cannot be opened!\n");
		return;
	}

	// get number of rows, columns, non zero values 
	fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);

	int* row_ptr_t = (int*)malloc((*num_rows + 1) * sizeof(int));
	int* col_ind_t = (int*)malloc(*num_vals * sizeof(int));
	int* row_ids_t = (int*)malloc(*num_vals * sizeof(int));
	int* col_ids_t = (int*)malloc(*num_vals * sizeof(int));
	float* values_t = (float*)malloc(*num_vals * sizeof(float));

	// collect occurrances of each row for determining the indices of row_ptr
	int* row_occurrances = (int*)malloc(*num_rows * sizeof(int));
	for (int i = 0; i < *num_rows; i++) {
		row_occurrances[i] = 0;
	}

	int row, column;
	float value;
	int idx = 0; 
	while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
		// subtract 1 from row and column indices to match C format 
		row--;
		column--;
		row_ids_t[idx] = row;
		col_ids_t[idx] = column;
		row_occurrances[row]++;
		idx++;
	}

	// set row_ptr
	int index = 0;
	for (int i = 0; i < *num_rows; i++) {
		row_ptr_t[i] = index;
		index += row_occurrances[i];
	}
	row_ptr_t[*num_rows] = *num_vals;
	free(row_occurrances);

	// set the file position to the beginning of the file 
	rewind(file);

	// read the file again, save column indices and values 
	for (int i = 0; i < *num_vals; i++) {
		col_ind_t[i] = -1;
	}

	fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals);
	int i = 0;
	while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
		row--;
		column--;
		// find the correct index (i + row_ptr_t[row]) using both row information and an index i 
		while (col_ind_t[i + row_ptr_t[row]] != -1) {
			i++;
		}
		col_ind_t[i + row_ptr_t[row]] = column;
		values_t[i + row_ptr_t[row]] = value;
		i = 0;
	}

	fclose(file);

	*row_ptr = row_ptr_t;
	*col_ind = col_ind_t;
	*values = values_t;
	*row_ids = row_ids_t;
	*col_ids = col_ids_t;
}