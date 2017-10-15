/*
 * ����CUDA����ı���
 *
 * cuda_sample_2.cu ּ�ڼ������ִ�������ĵ�GPUʱ������
 * ���ĵ�ʱ�� = ���ĵ�ʱ������/GPUʱ��Ƶ��
 * �Դ���� = ������ / ��ʱ
 * 
 * @author chenyang li
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 1024*1024

void printDeviceProps(const cudaDeviceProp *prop);

int data[DATA_SIZE];
// �豸ʱ��Ƶ��
int clockRate;

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

/* ����ƽ���ͣ�__global__����������GPU��*/
__global__ static void sumOfSquares(int *numbers, int *result, clock_t *time) {
	int sum, i;
	clock_t start, end;

	// ��ȡ��ʼʱ��
	start = clock();

	sum = 0;
	for (i = 0; i < DATA_SIZE; i++) {
		sum += numbers[i] * numbers[i];
	}
	*result = sum;

	// ��ȡ����ʱ��
	end = clock();

	*time = end - start;
}

int main(void) {
	if (!initCUDA()) {
		return 0;
	}

	int *gpudata, *result;
	int i, sum;
	clock_t time_used;
	clock_t *gpu_time_used;

	generateNumbers(data, DATA_SIZE);

	cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));
	// ���Դ��Ϸ���clock_t���������ڴ�ռ�
	cudaMalloc((void**)&gpu_time_used, sizeof(clock_t));

	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	sumOfSquares << < 1, 1, 0 >> > (gpudata, result, gpu_time_used);

	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, gpu_time_used, sizeof(clock_t), cudaMemcpyDeviceToHost);

	cudaFree(gpudata);
	cudaFree(result);
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