#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "schoolbook/helloworld.cuh"
#include <stdio.h>
__global__ void helloFromGPU(void) {
    printf("hello world from GPU\n");
}