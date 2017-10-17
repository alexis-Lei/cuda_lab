/*
 * matrix_mulitiplication.cu 并行矩阵乘法
 *
 * 假设两个待相乘矩阵均为方块矩阵（MATRIX_SIZE * MATRIX_SIZE） 
 * 
 * @author chenyang li
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define THREAD_NUM 256
// 矩阵大小为 MATRIX_SIZE * MATRIX_SIZE
#define MATRIX_SIZE 1000

// 将数组定义为全局，避免在栈内分配大块内存
float A[MATRIX_SIZE*MATRIX_SIZE], B[MATRIX_SIZE * MATRIX_SIZE], C[MATRIX_SIZE * MATRIX_SIZE];
float *gpu_A, *gpu_B, *gpu_C;

/* 打印GPU设备信息 */
void printDeviceProps(const cudaDeviceProp *prop) {
    printf("Device Name: %s\n", prop->name);
    printf("totalGlobalMem: %ld\n", prop->totalGlobalMem);
    printf("sharedMemPerBlock: %d\n", prop->sharedMemPerBlock);
    printf("regsPerBlock: %d\n", prop->regsPerBlock);
    printf("warpSize: %d\n", prop->warpSize);
    printf("memPitch: %d\n", prop->memPitch);
    printf("maxThreadPerBlock: %d\n", prop->maxThreadsPerBlock);
    printf("maxThreadsDim[0-2]: %d %d %d\n", prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);
    printf("maxGridSize[0-2]: %d %d %d\n", prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);
    printf("totalConstMem: %d\n", prop->totalConstMem);
    printf("major: %d & minor: %d\n", prop->major, prop->minor);
    printf("clockRate: %d\n", prop->clockRate);
    printf("textureAlignment: %d\n", prop->textureAlignment);
    printf("deviceOverlap: %d\n", prop->deviceOverlap);
    printf("multiProcessorCount: %d\n", prop->multiProcessorCount);
}

/* CUDA 初始化 */
bool initCUDA() {
    int count, i;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);

    if (0 == count) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for (i = 0; i < count; i++) {
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    cudaSetDevice(i);

    printDeviceProps(&prop);

    return true;
}

/* 构造二维数组，使用一维数组存储 */
void generateMatrix(float *mat, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            mat[i * size + j] = rand() % 10;
        }
    }
}

/* 打印矩阵 */
void printMatrix(float *mat, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf("%f ", mat[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// kernel函数，实现矩阵乘法
__global__ static void matrixMultiplication(const float *A, const float *B, float *C, int size) {
    // 当前线程所在Block的编号（从0开始）
    const int block_id = blockIdx.x;
    //当期线程编号（从0开始）
    const int thread_id = threadIdx.x;
    int i;
    int index, row, column;
    float s;

    // 当前线程全局索引（非Block内的线程索引）
    index = block_id * THREAD_NUM + thread_id;
    
    /* 当前线程将计算C[row][column] */
    row = index / size;
    column = index % size;

    s = 0.0f;
    if (row < size && column < size) {
        // A[row][0], A[row][1], A[row][2] ... A[row][size]
        // B[0]column], B[1][column], B[2][column] ... B[size][column]
        for (i = 0; i < size; i++) {
            s += A[row * size + i] * B[i * size + column];
        }
        C[row * size + column] = s;
    }
}

int main(void) {
    if (!initCUDA()) {
        return 0;
    }

    const int block_num = (MATRIX_SIZE * MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;
    int i, j;

    /* 构建矩阵 */
    generateMatrix(A, MATRIX_SIZE);
    generateMatrix(B, MATRIX_SIZE);

    /* 分配显存 */
    cudaMalloc((void**)&gpu_A, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMalloc((void**)&gpu_B, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMalloc((void**)&gpu_C, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

    /* 将数组从主内存拷贝至显存 */
    cudaMemcpy(gpu_A, A, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);

    // 执行kernel函数
    matrixMultiplication << <block_num, THREAD_NUM, 0 >> > (gpu_A, gpu_B, gpu_C, MATRIX_SIZE);

    // 将数组从显存拷贝至主内存
    cudaMemcpy(C, gpu_C, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyDeviceToHost);

    /* 释放显存空间 */
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);

    /* Optional */
    // printMatrix(A, MATRIX_SIZE);
    // printMatrix(B, MATRIX_SIZE);
    // printMatrix(C, MATRIX_SIZE);

    system("pause");

    // return 0;
}