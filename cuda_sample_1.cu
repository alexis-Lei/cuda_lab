/*
 * 计算数组元素平方和
 * 
 * cuda_sample_1.cu 仅实现在GPU上的计算，暂未涉及并行.
 *
 * @author chenyang li
 */
#include <stdio.h>
#include <stdlib.h>
// CUDA Runtime API
#include <cuda_runtime.h>

#define DATA_SIZE 1024*1024

int data[DATA_SIZE];

/* 产生0-9之间的随机数 */
void generateNumbers(int *numbers, int size) {
    int i;
    for (i = 0; i < size; i++) {
        numbers[i] = rand() % 10;
    }
}

/* CUDA 初始化 */
bool initCUDA() {
    int count, i;
    // 取得支持CUDA的装置的数目
    cudaGetDeviceCount(&count);

    if (0 == count) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
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

    return true;
}

/* 计算平方和（__global__函数运行于GPU）*/
__global__ static void sumOfSquares(int *numbers, int *result) {
    int sum, i;
    sum = 0;
    for (i = 0; i < DATA_SIZE; i++) {
        sum += numbers[i] * numbers[i];
    }
    *result = sum;
}

int main(void) {
    if (!initCUDA()) {
        return 0;
    }

    int *gpudata, *result;
    int i, sum;

    generateNumbers(data, DATA_SIZE);

    // 在显存上分配空间
    // 思考：为什么cudaMalloc函数原型的第一个参数类型为 (void **)？
    // 原因：gpudata指向某块内存区域的首地址，cudaMalloc在显存中分配一块内存，然后将该内存区域的首地址
    //      赋值给gpudata，因此cudaMalloc修改的是gpudata本身的值，而不是gpudata指向的内存区域的值。
    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));

    // 将数据从内存复制到显存
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // 执行kernel函数，语法：函数名<<<block数, thread数, share memory大小>>>
    sumOfSquares << < 1, 1, 0 >> > (gpudata, result);

    // 把计算结果从显存复制到内存
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpudata);
    cudaFree(result);

    printf("GPU sum is: %d\n", sum);

    sum = 0;
    for (i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("CPU sum is: %d\n", sum);

    system("pause");

    // return 0;
}