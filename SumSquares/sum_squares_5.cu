/*
 * sum_squares_5.cu 在改进存取模式的基础上，利用Block增加线程数量，进一步优化数组元素平方和计算
 *
 * @author chenyang li
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define DATA_SIZE 1024 * 1024
#define THREAD_NUM 216
// Block数量
#define BLOCK_NUM 32

int data[DATA_SIZE];
int clockRate;

/* 产生0-9之间的随机数 */
void generateNumbers(int *numbers, int size) {
    int i;
    for (i = 0; i < size; i++) {
        numbers[i] = rand() % 10;
    }
}

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
    printf("clockRate: %d\n", prop->clockRate); clockRate = prop->clockRate;
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

/* 寻找耗时最大的元素 */
clock_t findMaxTime(clock_t *time, int size) {
    int i;
    clock_t max = time[0];
    for (i = 0; i < size; i++) {
        if (time[i] > max) {
            max = time[i];
        }
    }
    return max;
}

/* 计算平方和（__global__函数运行于GPU）*/
__global__ static void sumOfSquares(int *numbers, int *sub_sum, clock_t *time) {
    int i;
    clock_t start, end;

    // 获取当前线程所属的Block号（从0开始）
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if (thread_id == 0) {
        start = clock();
    }

    sub_sum[block_id * THREAD_NUM + thread_id] = 0;
    // Block0-线程0获取第0个元素，Block0-线程1获取第1个元素...Block1-线程0获取第THREAD_NUM个元素，以此类推... 
    for (i = block_id * THREAD_NUM + thread_id; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        sub_sum[block_id * THREAD_NUM + thread_id] += numbers[i] * numbers[i];
    }

    if (thread_id == 0) {
        end = clock();
        time[block_id] = end - start;
    }
}

int main(void) {
    if (!initCUDA()) {
        return 0;
    }

    int *gpudata;
    int i, sum;
    int sub_sum[BLOCK_NUM * THREAD_NUM], *gpu_sub_sum;
    // 每个Block设置一个计时单元
    clock_t time_used[BLOCK_NUM], *gpu_time_used;

    generateNumbers(data, DATA_SIZE);

    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    // 当前一共有BLOCK_NUM * THREAD_NUM个线程
    cudaMalloc((void**)&gpu_sub_sum, sizeof(int) * BLOCK_NUM * THREAD_NUM);
    cudaMalloc((void**)&gpu_time_used, sizeof(clock_t) * BLOCK_NUM);

    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // 更新Block数
    sumOfSquares << < BLOCK_NUM, THREAD_NUM, 0 >> > (gpudata, gpu_sub_sum, gpu_time_used);

    cudaMemcpy(time_used, gpu_time_used, sizeof(clock_t) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_sum, gpu_sub_sum, sizeof(int) * BLOCK_NUM * THREAD_NUM, cudaMemcpyDeviceToHost);

    sum = 0;
    for (i = 0; i < BLOCK_NUM * THREAD_NUM; i++) {
        sum += sub_sum[i];
    }

    cudaFree(gpudata);
    cudaFree(gpu_sub_sum);
    cudaFree(time);

    clock_t max_time_used = findMaxTime(time_used, BLOCK_NUM);
    printf("\nGPU sum is: %d, time used: %f (s)\n", sum, (float)max_time_used / (clockRate * 1000));

    sum = 0;
    for (i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("CPU sum is: %d\n", sum);
    printf("Memory bandwidth: %f (MB/s)\n", ((float)(DATA_SIZE * sizeof(int) / 1024 / 1024)) / ((float)max_time_used / (clockRate * 1000)));

    system("pause");

    // return 0;
}