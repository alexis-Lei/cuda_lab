/*
 * cuda_sample_3.cu �������л�����Ԫ��ƽ���ͼ���
 * ÿ���̸߳����ۼ������е�һ����������Ԫ��
 *
 * @author chenyang li
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// ���� threadIdx
#include <device_launch_parameters.h>
#include <time.h>

#define DATA_SIZE 1024*1024

// �߳���
#define THREAD_NUM 256

int data[DATA_SIZE];
int clockRate;

/* ����0-9֮�������� */
void generateNumbers(int *numbers, int size) {
    int i;
    for (i = 0; i < size; i++) {
        numbers[i] = rand() % 10;
    }
}

/* ��ӡGPU�豸��Ϣ */
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

/* CUDA ��ʼ�� */
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

/* ����ƽ���ͣ�__global__����������GPU��*/
__global__ static void sumOfSquares(int *numbers, int *sub_sum, clock_t *time) {
    int i;
    clock_t start, end;

    // ��ȡ��ǰ�߳�Id����0��ʼ��
    const int thread_id = threadIdx.x;
    // ÿ���߳��ۼ�Ԫ�صĸ���
    const int size = DATA_SIZE / THREAD_NUM;

    // ��¼�߳�0����ʼʱ��
    if (thread_id == 0) {
        start = clock();
    }

    sub_sum[thread_id] = 0;
    for (i = thread_id * size; i < (thread_id + 1) * size; i++) {
        sub_sum[thread_id] += numbers[i] * numbers[i];
    }

    // ��¼�߳�0�Ľ���ʱ��
    if (thread_id == 0) {
        end = clock();
        *time = end - start;
    }
}

int main(void) {
    if (!initCUDA()) {
        return 0;
    }

    int *gpudata;
    int i, sum;
    int sub_sum[THREAD_NUM], *gpu_sub_sum;
    clock_t time_used, *gpu_time_used;

    generateNumbers(data, DATA_SIZE);

    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    // ʹ�ó���ΪTHREAD_NUM����������¼ÿ���̼߳���Ľ��
    cudaMalloc((void**)&gpu_sub_sum, sizeof(int) * THREAD_NUM);
    cudaMalloc((void**)&gpu_time_used, sizeof(clock_t));

    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // �����߳�����
    sumOfSquares << < 1, THREAD_NUM, 0 >> > (gpudata, gpu_sub_sum, gpu_time_used);

    cudaMemcpy(&time_used, gpu_time_used, sizeof(clock_t), cudaMemcpyDeviceToHost);
    // ���Դ��е����鿽�������ڴ���
    cudaMemcpy(sub_sum, gpu_sub_sum, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);

    sum = 0;
    for (i = 0; i < THREAD_NUM; i++) {
        sum += sub_sum[i];
    }

    cudaFree(gpudata);
    // �ͷ��Դ��е�����
    cudaFree(gpu_sub_sum);
    cudaFree(time);

    printf("\nGPU sum is: %d, time used: %f (s)\n", sum, (float)time_used / (clockRate * 1000));

    sum = 0;
    for (i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("CPU sum is: %d\n", sum);
    printf("Memory bandwidth: %f (MB/s)\n", ((float)(DATA_SIZE * sizeof(int) / 1024 / 1024)) / ((float)time_used / (clockRate * 1000)));

    system("pause");

    // return 0;
}