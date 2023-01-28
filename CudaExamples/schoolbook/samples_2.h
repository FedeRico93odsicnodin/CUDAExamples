/*
    examples from the first 3 chapters of the book 
*/
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../common.h"

// EXAMPLE 15: GLOBAL VARIABLE 
/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */
__device__ float devData;
__global__ void checkGlobalVariable() {
    // display the original value 
    printf("Device: the value of the global variable is %f\n", devData);

    // alter the value 
    devData += 2.0f;
}
void globalVariableDeclarationAndModification() {
    // initialize the global variable
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host: copied %f to the global variable\n", value);

    // invoke the kernel 
    checkGlobalVariable << <1, 1>> > ();

    // copy the global variable back to the host 
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
}

// EXAMPLE 16: SIMPLE MEMORY TRANSFER 
/*
 * An example of using CUDA's memory copy API to transfer data to and from the
 * device. In this case, cudaMalloc is used to allocate memory on the GPU and
 * cudaMemcpy is used to transfer the contents of host memory to an array
 * allocated using cudaMalloc.
 */
void simplemMemTransfer() {
    // set up the device 
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information 
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("starting at ");
    printf("device %d: %s memory size %d nbyte %5.2MB\n", 
        dev, deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    // allocate the host memory 
    float* h_a = (float*)malloc(nbytes);

    // allocate the device memory 
    float* d_a;
    CHECK(cudaMalloc((float**)&d_a, nbytes));

    // initialize the host memory 
    for (unsigned int i = 0; i < isize; i++) h_a[i] = 0.5f;

    // transfer data from the host to the device 
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host 
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory 
    CHECK(cudaFree(d_a));
    free(h_a);

    // reset the device 
    CHECK(cudaDeviceReset());
}

// EXAMPLE 17: SUM ARRAYS IN 0 MEMORY 
/*
 * This example demonstrates the use of zero-copy memory to remove the need to
 * explicitly issue a memcpy operation between the host and device. By mapping
 * host, page-locked memory into the device's address space, the address can
 * directly reference a host array and transfer its contents over the PCIe bus.
 *
 * This example compares performing a vector addition with and without zero-copy
 * memory.
 */
__global__ void sumArrays(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
void sumArraysZeroCopy() {
    // set up the device 
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // get device properties
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // check if support mapped memory 
    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        return;
    }

    printf("Using Device %d: %s", dev, deviceProp.name);

    // set up data size of vectors
    int ipower = 10;

    // TODO: setting as input
    //if (argc > 1) ipower = atoi(argv[1]);

    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

    if (ipower < 18) {
        printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, ipower, (float)nBytes / (1024.0f));
    }
    else {
        printf("Vector size %d power %d nbytes %3.0f MB\n", nElem, ipower, (float)nBytes / (1024.0f * 1024.0f));
    }

    // part 1: using device memory 
    // malloc host memory 
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side 
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks 
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory 
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device 
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // set up execution configuration 
    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArrays << <grid, block >> > (d_A, d_B, d_C, nElem);

    // copy kernel result back to host side 
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results 
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory 
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    // free host memory 
    free(h_A);
    free(h_B);


    // part 2: using zerocopy memory for array A and B 
    // allocate zerocpy memory 
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side 
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // pass the pointer to the device 
    CHECK(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
    CHECK(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));

    // add at host side for result checks 
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory 
    sumArraysZeroCopy << <grid, block >> > (d_A, d_B, d_C, nElem);

    // copy kernel result back to host side 
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results 
    checkResult(hostRef, gpuRef, nElem);

    // free memory 
    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));

    free(hostRef);
    free(gpuRef);

    // reset device 
    CHECK(cudaDeviceReset());

}

/*
    this example demostrates the impact of misaligned reads on performance by 
    forcing misaligned reads to occur on a float *
*/
void sumArraysOnHost(float* A, float* B, float* C, const int n, int offset) {
    for (int idx = offset, k = 0; idx < n; idx++, k++) {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float* A, float* B, float* C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float* A, float* B, float* C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

void readSegment() {
    int offset = 0;
    int blocksize = 512;
    // set up device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // set up the array size 
    int nElem = 1 << 20; // total number of element to reduce 
    printf("with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // asking for the offset and for the blocksize 
    printf("give the number for which starting the reduction (try for instance 11 for having a misaligned offset)");
    scanf(" %d", &offset);
    printf("give the blocksize preferred (default 512)");
    scanf(" %d", &blocksize);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // allocation of the host memory 
    float* h_A = (float*)malloc(nBytes);
    float* h_B = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    // initialize host array 
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // summary at host side 
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory 
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device 
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    // kernel 1:
    double iStart = seconds();
    warmup << <grid, block >> > (d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup <<<%4d, %4d>>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    iStart = seconds();
    readOffset << <grid, block >> > (d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("readOffset <<<%4d, %4d>>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results 
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // free host and device memory 
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device 
    CHECK(cudaDeviceReset());
    return;
}

/*
    A simple example of using an array of structures to store data on the device 
    This example is used to study the impact on performance of data layout on the 
    GPU.

    AoS: one contiguous 64-bit read to get x and y (up to 300 cycles)
*/

#define LEN 1<<22

struct innerStruct {
    float x;
    float y;
};

struct innerArray {
    float x[LEN];
    float y[LEN];
};

void initialInnerStruct(innerStruct* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void testInnerStructHost(innerStruct* A, innerStruct* C, const int n) {
    for (int idx = 0; idx < n; idx++) {
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
    }

    return;
}

void checkInnerStruct(innerStruct* hostRef, innerStruct* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].x, gpuRef[i].x);
            break;
        }
        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].y, gpuRef[i].y);
            break;
        }
    }

    if (!match) printf("Arrays do not match.\n\n");
}

__global__ void testInnerStruct(innerStruct* data, innerStruct* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

__global__ void warmup(innerStruct* data, innerStruct* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

void simpleMathAoS() {
    // set up the device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // allocate host memory 
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct* h_A = (innerStruct*)malloc(nBytes);
    innerStruct* hostRef = (innerStruct*)malloc(nBytes);
    innerStruct* gpuRef = (innerStruct*)malloc(nBytes);

    // initialize host array 
    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    // allocate device memory 
    innerStruct* d_A, * d_C;
    CHECK(cudaMalloc((innerStruct**)&d_A, nBytes));
    CHECK(cudaMalloc((innerStruct**)&d_C, nBytes));

    // copy data from host to device 
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up offset for sommaryAU: it is blocksize not offset. Thanks.CZ
    int blocksize = 128;
    int selectionBlockSize = 0;
    printf("select the preferred blocksize (0 for default 128)");
    scanf(" %d", &selectionBlockSize);
    if (selectionBlockSize > 0)
        blocksize = selectionBlockSize;

    // execution configuration 
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // kernel 1: warmup
    double iStart = seconds();
    warmup << <grid, block >> > (d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup          <<<%3d, %3d>>> elapsed %f sec\n", grid.x, block.x, iElaps);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // kernel 2: test inner struct 
    iStart = seconds();
    testInnerStruct << <grid, block >> > (d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("innestruct<<<%3d, %3d>>> elapsed %f sec\n", grid.x, block.x, iElaps);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // free memories both host and device 
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device 
    CHECK(cudaDeviceReset());
    return;
}
