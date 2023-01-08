#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "schoolbook/samples_1.h"
#include "schoolbook/samples_2.h"
#include <stdio.h>
#include<conio.h>
#include "../CudaExamples/slides/01_prefixsum.h";
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
        printf("200_prefixsum");
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
            bool en = false;
            int temp;
            printf("enable console? (0-1)");
            scanf("%d", &temp);
            en = temp;
            prefixSumFirstVersion(en);
            break;
        }
        printf("press a key to continue");
        getch();
    } while (selection > 0);
    return 0;
}
