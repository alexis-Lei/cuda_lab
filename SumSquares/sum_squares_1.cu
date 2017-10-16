/*
 * ��������Ԫ��ƽ����
 * 
 * cuda_sample_1.cu ��ʵ����GPU�ϵļ��㣬��δ�漰����.
 *
 * @author chenyang li
 */
#include <stdio.h>
#include <stdlib.h>
// CUDA Runtime API
#include <cuda_runtime.h>

#define DATA_SIZE 1024*1024

int data[DATA_SIZE];

/* ����0-9֮�������� */
void generateNumbers(int *numbers, int size) {
    int i;
    for (i = 0; i < size; i++) {
        numbers[i] = rand() % 10;
    }
}

/* CUDA ��ʼ�� */
bool initCUDA() {
    int count, i;
    // ȡ��֧��CUDA��װ�õ���Ŀ
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

/* ����ƽ���ͣ�__global__����������GPU��*/
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

    // ���Դ��Ϸ���ռ�
    // ˼����ΪʲôcudaMalloc����ԭ�͵ĵ�һ����������Ϊ (void **)��
    // ԭ��gpudataָ��ĳ���ڴ�������׵�ַ��cudaMalloc���Դ��з���һ���ڴ棬Ȼ�󽫸��ڴ�������׵�ַ
    //      ��ֵ��gpudata�����cudaMalloc�޸ĵ���gpudata�����ֵ��������gpudataָ����ڴ������ֵ��
    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));

    // �����ݴ��ڴ渴�Ƶ��Դ�
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // ִ��kernel�������﷨��������<<<block��, thread��, share memory��С>>>
    sumOfSquares << < 1, 1, 0 >> > (gpudata, result);

    // �Ѽ��������Դ渴�Ƶ��ڴ�
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