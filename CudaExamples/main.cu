#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "schoolbook/samples_1.h"
#include "schoolbook/samples_2.h"
#include <stdio.h>
#include<conio.h>
#include "../CudaExamples/slides/01_prefixsum.h";
#include "../CudaExamples/slides/02_mergesort.h";
#include "../CudaExamples/slides/03_sharedmemory.h";
#include "../CudaExamples/slides/04_histograms.h";
// permette di stabilire se abilitare o meno la modalità console per alcuni degli esempi
int main()
{
    int selection;
    do {
        printf("\n\n");
        printf("SELECT THE EXAMPLE TO RUN:\n");
        printf("1_helloworld\n");
        printf("2_checkDimension\n");
        printf("3_defineGridBlock\n");
        printf("4_sumArraysOnGPUSmallCase\n");
        printf("5_sumArraysOnGPUTimer\n");
        printf("6_checkThreadIndex\n");
        printf("7_matrixSummationOnGPU2DGrid2DBlock\n");
        printf("8_checkDeviceInfo\n");
        printf("9_determiningTheBestGPU\n");
        printf("10_simpleWarpDivergence\n");
        printf("11_reduceInteger\n");
        printf("12_nestedHelloWorld\n");
        printf("13_nestedReduce\n");
        printf("14_nestedReduce2 (to check)\n");
        printf("-----------\n");
        printf("15_globalVariableDeclarationAndModification\n");
        printf("16_simplemMemTransfer\n");
        printf("17_sumArraysZeroCopy\n");
        printf("\n---- SLIDES EXAMPLES ----\n");
        printf("200_prefixsum\n");
        printf("201_prefixsum (second version)\n");
        printf("202_prefixsum (third version)\n");
        printf("203_prefixsum (fourth version)\n");
        printf("204_prefixsum (fifth version)\n");
        printf("-----------\n");
        printf("205_mergesort\n");
        printf("206_mergesort (rank single block)\n");
        printf("-----------\n");
        printf("207_parallelReductionShared\n");
        printf("208_parallelReductionSharedOptimized\n");
        printf("209_parallelReductionSharedOptimizedWarp\n");
        printf("210_mergeSortTiledShared (merge sort using the shared memory)\n");
        printf("-----------\n");
        printf("211_histograms on GPU\n");
        printf("212_histograms on GPU (Coalesced) - (TO CHECK)\n");
        printf("213_histograms on GPU (Coalesced - Privatize bins)\n");
        printf("\n");
        scanf("%d", &selection);
        printf("\n");
        if (selection == 0)
            break;
        switch (selection) {
        case 1:
            helloWorld();
            break;
        case 2:
            checkDimension();
            break;
        case 3:
            defineGridBlock();
            break;
        case 4: 
            sumArraysOnGPUSmallCase();
            break;
        case 5: 
            sumArraysOnGPUTimer();
            break;
        case 6:
            checkThreadIndex();
            break;
        case 7:
            matrixSummationOnGPU2DGrid2DBlock();
            break;
        case 8: 
            checkDeviceInfo();
            break;
        case 9: 
            determiningTheBestGPU();
            break;
        case 10:
            simpleWarpDivergence();
            break;
        case 11: 
            reduceInteger();
            break;
        case 12: 
            nestedHelloWorld();
            break;
        case 13: 
            nestedReduce();
            break;
        case 14: 
            nestedReduce2();
            break;
        case 15:
            globalVariableDeclarationAndModification();
            break;
        case 16: 
            simplemMemTransfer();
            break;
        case 17:
            sumArraysZeroCopy();
            break;
        case 200:
            prefixSumFirstVersion();
            break;
        case 201: 
            prefixSumSecondVersion();
            break;
        case 202: 
            prefixSumThirdVersion();
            break;
        case 203:
            prefixSumFourthVersion();
            break;
        case 204: 
            prefixSumFifthVersion();
            break;
        case 205: 
            sequentialMergeSort();
            break;
        case 206: 
            mergeSortRankBlock();
            break;
        case 207:
            parallelReductionShared();
            break;
        case 208:
            parallelReductionSharedOptimized();
            break;
        case 209:
            parallelReductionSharedOptimizedWarp();
            break;
        case 210:
            mergeSortTiledShared();
            break;
        case 211: 
            histogramGPU();
            break;
        case 212: 
            histogramGPUCoalesced();
            break;
        case 213:
            histogramGPUCoalescedPrivate();
            break;
        }
        printf("press a key to continue");
        getch();
    } while (selection > 0);
    return 0;
}
