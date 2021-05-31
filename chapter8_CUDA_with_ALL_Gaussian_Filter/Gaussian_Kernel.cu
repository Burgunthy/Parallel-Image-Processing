#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

extern "C" void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst,
									int w, int h, float *cuGkernel, int kernel_size);

__global__
void cuda_Filter_2D(uchar * pSrcImage, uchar *pDstImage,
					int SrcWidth, int SrcHeight, float *pKernel, int KWidth, int KHeight)
{
	// 블록과 쓰레드 주소에 따라 현재 pixel의 index를 계산한다 
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int mSize = KWidth / 2;

	float temp = 0.f;

	// Serial과 동일하게 예외처리 진행 후 각 pixel 계산
	if (x >= KWidth / 2 && y >= KHeight / 2
		&& x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int j = -mSize; j <= mSize; j++) {
			for (int i = -mSize; i <= mSize; i++) {

				// float 형태로 계산 및 저장
				temp += (float)pSrcImage[index + i + j * SrcWidth]
					* pKernel[i + mSize + (j + mSize) * KHeight];
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
	dim3 grid = dim3(w / 16, h / 16);
	dim3 block = dim3(16, 16);

	// 각 pixel 별 CUDA Gaussain Filter 진행
	cuda_Filter_2D <<< grid, block >>> (pcuSrc, pcuDst, w, h, cuGkernel, kernel_size, kernel_size);

	// 메모리 싱크로나이즈 진행
	cudaThreadSynchronize();
}