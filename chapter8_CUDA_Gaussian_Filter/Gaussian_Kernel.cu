#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

extern "C" void Gaussian_Filter_2D(float *pcuSrc, float *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

__global__ void cuda_Filter_2D(float * pSrcImage, int SrcWidth, int SrcHeight, float *pKernel, int KWidth, int KHeight, float *pDstImage) {
	
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int mSize = KWidth / 2;

	float temp;

	if (x >= KWidth / 2 && y >= KHeight / 2 && x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		temp = 0;
		for (int j = -mSize; j <= mSize; j++) {
			for (int i = -mSize; i <= mSize; i++) {
				temp += pSrcImage[index + i + j * SrcWidth] * pKernel[i + mSize + (j + mSize) * KHeight];
			}
		}
		pDstImage[index] = temp;
	}
	else
	{
		pDstImage[index] = 0;		// 예외 처리 진행?
	}
}

void Gaussian_Filter_2D(float *pcuSrc, float *pcuDst, int w, int h, float *cuGkernel, int kernel_size) {

	dim3 grid = dim3(w / 16, h / 16);
	dim3 block = dim3(16, 16);

	//float *pcuBuf;
	//(cudaMalloc((void**)&pcuBuf, w*h * sizeof(float)));

	cuda_Filter_2D << < grid, block >> > (pcuSrc, w, h, cuGkernel, kernel_size, kernel_size, pcuDst);
	
	cudaThreadSynchronize();

	/*float *PrintKernel = new float[kernel_size*kernel_size];
	cudaMemcpy(PrintKernel, cuGkernel, kernel_size*kernel_size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++)
		{
			printf("%f\t", PrintKernel[i*kernel_size + j]);
		}
		printf("\n");
	}*/
}