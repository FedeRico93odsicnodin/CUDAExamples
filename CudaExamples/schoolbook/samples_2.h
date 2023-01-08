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