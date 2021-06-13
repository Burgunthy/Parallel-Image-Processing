#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

extern "C" void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

__constant__ float constKernel[5 * 5];

__global__
void cuda_constant_Filter_2D(uchar * pSrcImage, uchar *pDstImage,
	int SrcWidth, int SrcHeight, int KWidth, int KHeight)
{
	// 블록과 쓰레드 주소에 따라 현재 pixel의 index를 계산한다
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;

	int index = y * SrcWidth * 3 + x * 3 + tz;
	int mSize = KWidth / 2;

	float temp = 0.f;

	// Serial과 동일하게 예외처리 진행 후 각 pixel 계산
	if (x >= KWidth / 2 && y >= KHeight / 2
		&& x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int j = -mSize; j <= mSize; j++) {
			for (int i = -mSize; i <= mSize; i++) {

				// float 형태로 계산 및 저장
				temp += (float)pSrcImage[index + i * 3 + j * SrcWidth * 3]
					* constKernel[i + mSize + (j + mSize) * KHeight];
			}
		}
		// 최종 dst 이미지에는 uchar의 형태로 저장
		pDstImage[index] = (uchar)temp;
	}
	else {
		pDstImage[index] = 0;		// kernel size 바깥의 픽셀의 예외처리 진행
	}
}

void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst,
	int w, int h, float *cuGkernel, int kernel_size) {
	// 16 x 16 쓰레드의 블록과 그 grid 사이즈 저장
	dim3 grid = dim3(w / 16, h / 16, 1);
	dim3 block = dim3(16, 16, 3);

	// 각 pixel 별 CUDA Gaussain Filter 진행

	cudaMemcpyToSymbol(constKernel, cuGkernel, sizeof(float)*kernel_size*kernel_size);
	cuda_constant_Filter_2D << < grid, block >> > (pcuSrc, pcuDst, w, h, kernel_size, kernel_size);

	// 메모리 싱크로나이즈 진행
	cudaThreadSynchronize();
}