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
#include "schoolbook/common.h"

// 01: simple hello world from GPU 
__global__ void helloFromGPU(void) {
    printf("hello world from GPU\n");
} 
void helloWorld() {
    printf("hello from CPU\n");
    helloFromGPU << <1, 10 >> > ();
    cudaDeviceReset();
}

// 02: check dimension
/* this examples aims to show the dimensions for thread - block - grid 
HOST and DEVICE side both
*/
__global__ void checkIndex(void) {
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
        "gridDim:(%d, %d, %d)\n", 
        threadIdx.x, threadIdx.y, threadIdx.z, 
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z
    );
}
void checkDimension() {
    // define total data element 
    int nElem = 6;

    // define grid and block structure 
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side 
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block,y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side 
    checkIndex << <grid, block >> > ();

    // reset device before you leave 
    cudaDeviceReset();
}

// 03 DEFINE GRID AND BLOCK 
/*
    this example shows how to initiliaze with some dimensions the 
    values for the grid and the block 
*/
void defineGridBlock() {
    // define total data elements 
    int nElem = 1024;

    // define grid and block structure 
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d grid.y %d \n", grid.x, block.x);

    // reset block 
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.y %d\n", grid.x, block.x);

    // reset block 
    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.y %d\n", grid.x, block.x);

    // reset block 
    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.y %d\n", grid.x, block.x);

    // reset device before you leave 
    cudaDeviceReset();
}

// 04 SUM ARRAY ON GPU SMALL CASE 
/*
    summing arrays on HOST and DEVICE and comparing the obtained results 
*/
// checking the result 
void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    printf("\n\n");
    if (match) printf("Arrays match.\n\n");
}
void initialData(float* ip, int size) {
    // generate different seed for random number 
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
void sumArraysOnHost(float* A, float* B, float* C, const int N) {
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}
__global__ void sumArraysOnGPU(float* A, float* B, float* C, bool printRes) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
    if(printRes)
        printf("threadIdx.x = %d - result of the sum: A[%d] + B[%d] = %f\n", i, i, i, C[i]);
}
// this is a modification of above kernel for summation on more grids and on the overall vector 
// TODO: eventuale implementazione per stampare il risultato 
__global__ void sumArraysOnGPU2(float* A, float* B, float* C, const int N, bool printRes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
    if (printRes)
        printf("threadIdx.x = %d - result of the sum: A[%d] + B[%d] = %f\n", i, i, i, C[i]);
}
void sumArraysOnGPUSmallCase() {
    // set up device 
    int dev = 0;
    cudaSetDevice(dev);

    // set up data size of vectors 
    int nElem = 32;
    printf("Vector size %d\n", nElem);

    // malloc host memory 
    size_t nBytes = nElem * sizeof(float);

    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host size 
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory 
    float* d_A, * d_B, * d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device 
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side 
    dim3 block(nElem);
    dim3 grid(nElem / block.x);
    printf("Execution configuration<<<%d, %d>>>\n", grid.x, block.x);
    sumArraysOnGPU << <grid, block >> > (d_A, d_B, d_C, true);
    printf("\n\n");
    
    // copy kernel result back to host side 
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result checks 
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results 
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory 
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}

// 05 SUM ARRAYS ON GPU WITH TIMER 
/*
    monitoring the performance of HOST and DEVICE arrays summation 
*/
void sumArraysOnGPUTimer() {
    // set up the device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // set up date size of vectors 
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory 
    size_t nBytes = nElem * sizeof(float);

    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side 
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks 
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory 
    float* d_A, * d_B, * d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device 
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side 
    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    iStart = seconds();
    sumArraysOnGPU2 << <grid, block >> > (d_A, d_B, d_C, nElem, false);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU<<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

    // copy kernel result back to host side 
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory 
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}

// 06 CHECK THREAD INDEX 
/*
    this example shows the different coordinates of a matrix of threads which 
    has been istantiated 
*/
void initialInt(int* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = i;
    }
}
void printMatrix(int* C, const int nx, const int ny) {
    int* ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%3d", ic[ix]);
        }

        ic += nx;
        printf("\n");
    }
    printf("\n");
}
__global__ void printThreadIndex(int* A, const int nx, const int ny) {

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf(
        "thread_id (%d,%d)" 
        " | block_id (%d,%d)" 
        " | coordinate (%d,%d) "
        " | global index %2d ival %2d\n", 
        threadIdx.x, threadIdx.y, 
        blockIdx.x, blockIdx.y, 
        ix, iy, 
        idx, A[idx]);
}
void checkThreadIndex() {
    // get device information 
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set matrix dimension 
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory 
    int* h_A;
    h_A = (int*)malloc(nBytes);

    // initialize host matrix with integer 
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // malloc device memory 
    int* d_MatA;
    cudaMalloc((void**)&d_MatA, nBytes);

    // transfer data from host to device 
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration 
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // printing the grid and block dimensions 
    printf("\nblock dimensions: %d, %d, %d", block.x, block.y, block.z);
    printf("\ngrid dimensions: %d, %d, %d", grid.x, grid.y, grid.z);

    printf("\n\nMONITORINIG THE THREADS INDEXES:\n\n");
    // invoke the kernel 
    printThreadIndex << < grid, block >> > (d_MatA, nx, ny);
    cudaDeviceSynchronize();

    // free host and device memory 
    cudaFree(d_MatA);
    free(h_A);

    // reset device 
    cudaDeviceReset();
}

// 07 MATRIX SUMMATION 
/*
    this example is a matrix summation on host and on device 
*/
void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny) {
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}
__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, int nx, int ny, bool enableConsole) {
    // thread identification in block / grid and global memory 
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    /*if (enableConsole) {
        printf("\n----------- THREAD EXECUTION ----------\n");
        printf("ix = %d, iy = %d, nx = %d, ny = %d\n", ix, iy, nx, ny);
        printf("idx = %d\n", idx);
    }*/
    // from each thread i map every block with its indices 
    if ((ix < nx) && (iy < ny)) {
        MatC[idx] = MatA[idx] + MatB[idx];
        if(enableConsole)
            printf("thread idx: %d - sum MatA[idx] + MatB[idx] = %f\n", idx, idx, idx, MatC[idx]);
    }
    //printf("\n\n");
}
void matrixSummationOnGPU2DGrid2DBlock() {
    // set up the device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // set up date size of matrix 
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory 
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side 
    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory 
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void**)&d_MatA, nBytes);
    cudaMalloc((void**)&d_MatB, nBytes);
    cudaMalloc((void**)&d_MatC, nBytes);

    // transfer data from host to device 
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side 
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = seconds();
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny, true);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("sumMatrixOnGPU2D<<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // copy kernel results back to host side 
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results 
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory 
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device 
    cudaDeviceReset();
}

// 08 CHECKING DEVICE INFORMATION 
/*
    this example shows how to query the device properties for getting some information 
*/
void checkDeviceInfo() {
    // detecting the number of devices that supports CUDA
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaDeviceCount returned %d\n -> %s\n",
            (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    // getting the properties for the first device 
    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\" \n", dev, deviceProp.name);

    // cuda driver and runtime version 
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("    CUDA Driver Version / Runtime Version           %d.%d / %d.%d\n",
        driverVersion / 1000, (driverVersion % 100) / 10,
        runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // cuda capability major / minor 
    printf("    CUDA Capability Major/Minor version number          %d.%d\n",
        deviceProp.major, deviceProp.minor);

    // total amount of global memory 
    printf("    Total amount of global memory:          %.2f Mbytes (%llu bytes)\n",
        (float)deviceProp.totalGlobalMem / (pow(1024.0, 3)),
        (unsigned long long) deviceProp.totalGlobalMem
    );

    // GPU clock rate 
    printf("    GPU clock rate:         %.0f MHz (%0.2f GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    // Memory clock rate 
    printf("    Memory clock rate:          %0.f MHz\n",
        deviceProp.memoryClockRate * 1e-3f);

    // Memory bus width 
    printf("    Memory bus width:           %d-bit\n",
        deviceProp.memoryBusWidth);

    // L2 cache size 
    if (deviceProp.l2CacheSize) {
        printf("    L2 Cache Size:          %d bytes\n",
            deviceProp.l2CacheSize);
    }

    // max texture dimension size (x.y.z)
    printf("    1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D,
        deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    // max layered texture size (dim) x layers 
    printf("    Max Layered texture size (dim) x layers 1D=(%d) x %d, 2D=(%d, %d) x %d\n",
        deviceProp.maxTexture1DLayered[0],
        deviceProp.maxTexture1DLayered[1],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

    // amount of constant memory 
    printf("    Total amount of constant memory:            %lu bytes\n",
        deviceProp.totalConstMem);

    // total amount of shared memory per block 
    printf("    Total amount of shared memory per block:            %lu bytes\n",
        deviceProp.sharedMemPerBlock);

    // total amount of registers available per block 
    printf("    total amount of registers available per block           %d\n",
        deviceProp.regsPerBlock);

    // warp size 
    printf("    warp size:          %d\n", deviceProp.warpSize);

    // maximum number of threads per multiprocessors 
    printf("    maximum number of threads per multiprocessor:           %d\n", deviceProp.maxThreadsPerMultiProcessor);

    // maximum number of threads per block 
    printf("    maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);

    // max sizes of each dimension of a block 
    printf("    Maximum sizes of each dimension of a block:         %d x %d x %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]
    );

    // max sizes of each dimension of a grid 
    printf("    Maximum sizes of each dimension of a grid:          %d x %d x %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]);

    // maximum memory pitch 
    printf("    Maximum memory pitch:           %lu bytes\n", deviceProp.memPitch);

}

// 09 DETERMINING THE BEST GPU 
/*
    this example shows the piece of code for determining the best GPU from the ones 
    available (in case there's only one GPU this is returned)
*/
void determiningTheBestGPU() {
    int numDevice = 0;
    cudaDeviceProp deviceProp;
    int selectedDevice = 0;
    // getting the device count for the current machine 
    cudaGetDeviceCount(&numDevice);
    if (numDevice > 1) {
        int maxMultiprocessor = 0, maxDevice = 0;
        for (int device = 0; device = numDevice; device++) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if (maxMultiprocessor < props.multiProcessorCount) {
                maxMultiprocessor = props.multiProcessorCount;
                maxDevice = device;
            }
        }
    }
    else {
        cudaSetDevice(selectedDevice);
        cudaGetDeviceProperties(&deviceProp, selectedDevice);
        printf("only device %s is available\n", deviceProp.name);
    }
}

// 10 SIMPLE WARP DIVERGENCE 
/*
    this example shows a warp divergence cases 
*/
// even and odd thread approach for thread partitioning 
// in warp divergence definition 
__global__ void MathKernel1(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
// avoiding warp divergence: the condition on the warp 
// forces the branch to be a multiple of warp size 
__global__ void mathKernel2(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    // condition for avoiding warp divergence 
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
// warming up function for reducing kernel overhead
__global__ void warmingup(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}
// math kernel 3: rewriting kernel1 to directly expose branch predication 
// in the kernel code 
__global__ void mathKernel3(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred) {
        ia = 100.0f;
    }

    if (!ipred) {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}
// math kernel 4
__global__ void mathKernel4(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0) {
        ia = 100.0f;
    }
    else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}
void simpleWarpDivergence() {
    // set up the device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    //printf("%s using device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size 
    int size = 64;
    int blocksize = 64; 
    // TODO: eventuale implementazione per ricevere parametri di input 
    /*if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);*/
    printf("Data size %d ", size);

    // set up execution configuration 
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float* d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    // run a warmup kernel to remove overhead
    size_t iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingup << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup <<<%4d, %4d>>> elapsed %d sec\n", grid.x, block.x, iElaps);

    // run kernel 1
    iStart = seconds();
    MathKernel1 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel1<<<%4d, %4d>>> elapsed %d sec\n", grid.x, block.x, iElaps);

    // run kernel 2
    iStart = seconds();
    mathKernel2 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel2<<<%4d, %4d>>> elapsed %d sec\n", grid.x, block.x, iElaps);

    // run kernel 3 
    iStart = seconds();
    mathKernel3 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel3<<<%4d, %4d>>> elapsed %d sec\n", grid.x, block.x, iElaps);

    // run kernel 4
    iStart = seconds();
    mathKernel4 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel4<<<%4d, %4d>>> elapsed %d sec\n", grid.x, block.x, iElaps);
    // free gpu memory and reset the device 
    cudaFree(d_C);
    cudaDeviceReset();
}

// 11 INTERLEAVED AND NEIGHBORED CUDA IMPLEMENTATIONS 
/*
    Implementation of the interleaved and neighbor-paired approaches to
    parallel reduction in CUDA.
    The sum operation is used in the example.
    A variety of optimizations on parallel reduction aimed at reducing divergence
    are also demostrated such as unrolling
*/
// RECURSIVE implementation of the Interleaved Pair Approach 
int recursiveReduce(int* data, int const size) {

    // terminate check 
    if (size == 1) return data[0];

    // renew the stride 
    int const stride = size / 2;

    // in-place reduction 
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively 
    return recursiveReduce(data, stride);
}
// Neighbored Pair Implementation with divergence 
__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert the global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check 
    if (idx >= n) return;

    // in-place reduction in global memory 
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride) == 0)) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadBlock 
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
// Neighbored Pair Implementation with less divergence 
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check 
    if (idx >= n) return;

    // in-place reduction in global memory 
    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        // convert tid into local array index 
        int index = 2 * stride * tid;

        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadBlock 
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
// Interleaved Pair Implementation with less divergence 
__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check 
    if (idx >= n) return;

    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global memory 
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnrolling2(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchonize within threadblock 
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnrolling4(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadBlock
        __syncthreads();
    }

    // write result for this block to global memory 
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnrolling8(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in globla memory 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock 
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnrollWarps8(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadBlock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to global memory 
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceCompleteUnrollWarps8(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll 
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global memory 
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll 
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp 
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global memory 
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnrollWarps(int* g_idata, int* g_odata, unsigned int n) {

    // set thread ID 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {

        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock 
        __syncthreads();
    }

    // unrolling last warp 
    if (tid < 32) {
        volatile int* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
void reduceInteger() {
    // set up device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    //printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization 
    int size = 1 << 24; // total number of elements to reduce 
    printf("    with array size %d", size);

    // execution configuration 
    int blocksize = 512; // intial block size 
    // TODO: eventuale implementazione per accettare parametri di input 
    //if (argc > 1) {
    //    blocksize = atoi(argv[1]); // block size from command line argument 
    //}

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory 
    size_t bytes = size * sizeof(int);
    int* h_idata = (int*)malloc(bytes);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // initialize the array 
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)(rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory 
    int* d_idata = NULL;
    int* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    // cpu reduction 
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored 
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored << <grid, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence 
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess << <grid, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d<<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved 
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved << <grid, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2 << <grid.x / 2, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4 << <grid.x / 4, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    // kernel 8: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8 << <grid.x / 8, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    switch (blocksize) {
    case 1024:
        reduceCompleteUnroll<1024> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 512:
        reduceCompleteUnroll<512> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 256:
        reduceCompleteUnroll<256> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 128:
        reduceCompleteUnroll<128> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    case 64:
        reduceCompleteUnroll<64> << <grid.x / 8, block >> > (d_idata, d_odata, size);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory 
    free(h_idata);
    free(h_odata);

    // free device memory 
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device 
    CHECK(cudaDeviceReset());

    // check the results 
    bResult = (gpu_sum == cpu_sum);
    if (!bResult) printf("test failed!\n");
}

// 12 NESTED REDUCE 
__global__ void gpuRecursiveReduce(int* g_idata, int* g_odata, unsigned int isize) {
    // set thread ID 
    unsigned int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int* odata = &g_odata[blockIdx.x];

    // stop condition 
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invocation 
    int istride = isize >> 1;

    if (istride > 1 && tid < istride) {
        // in place reduction 
        idata[tid] += idata[tid + istride];
    }

    // sync at block level 
    __syncthreads();

    // nested invocation to generate child grids
    if (tid == 0) {
        gpuRecursiveReduce << <1, istride >> > (idata, odata, istride);

        // sync all child grids launched in this block 
        cudaDeviceSynchronize();
    }

    // sync at block level again 
    __syncthreads();
}
int nestedReduce() {
    // set up device 
    int dev = 0, gpu_sum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("starting reduction at ");
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // set up execution configuration 
    int nblock = 2048;
    int nthread = 512;

    // TODO: this parameters as input 
    //if (argc > 1)
    //{
    //    nblock = atoi(argv[1]);   // block size from command line argument
    //}

    //if (argc > 2)
    //{
    //    nthread = atoi(argv[2]);   // block size from command line argument
    //}

    int size = nblock * nthread; // total number of elements to reduceNeighbored 

    dim3 block(nthread, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("array %d grid %d block %d\n\n", size, grid.x, block.x);

    // allocate host memory 
    size_t bytes = size * sizeof(int);
    int* h_idata = (int*)malloc(bytes);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    int* tmp = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        h_idata[i] = (int)(rand() & 0xFF);
        h_idata[i] = 1;
    }

    memcpy(tmp, h_idata, bytes);

    // allocate device memory 
    int* d_idata = NULL;
    int* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    double iStart, iElaps;

    // cpu recursive reduction 
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce\t\telapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // gpu reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    reduceNeighbored << <grid, block >> > (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // gpu nested reduce kernel 
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduce << <grid, block >> > (d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // free host memory 
    free(h_idata);
    free(h_odata);

    // free device memory 
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device 
    CHECK(cudaDeviceReset());

    // check the result 
    bResult = (gpu_sum == cpu_sum);

    if (!bResult) printf("test failted\n");
}