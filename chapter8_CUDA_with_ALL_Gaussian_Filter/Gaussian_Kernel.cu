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
	// ��ϰ� ������ �ּҿ� ���� ���� pixel�� index�� ����Ѵ� 
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int mSize = KWidth / 2;

	float temp = 0.f;

	// Serial�� �����ϰ� ����ó�� ���� �� �� pixel ���
	if (x >= KWidth / 2 && y >= KHeight / 2
		&& x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int j = -mSize; j <= mSize; j++) {
			for (int i = -mSize; i <= mSize; i++) {

				// float ���·� ��� �� ����
				temp += (float)pSrcImage[index + i + j * SrcWidth]
					* pKernel[i + mSize + (j + mSize) * KHeight];
			}
		}
		// ���� dst �̹������� uchar�� ���·� ����
		pDstImage[index] = (uchar)temp;
	}
	else {
		pDstImage[index] = 0;		// kernel size �ٱ��� �ȼ��� ����ó�� ����
	}
}

void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst,
						int w, int h, float *cuGkernel, int kernel_size) {
	// 16 x 16 �������� ��ϰ� �� grid ������ ����
	dim3 grid = dim3(w / 16, h / 16);
	dim3 block = dim3(16, 16);

	// �� pixel �� CUDA Gaussain Filter ����
	cuda_Filter_2D <<< grid, block >>> (pcuSrc, pcuDst, w, h, cuGkernel, kernel_size, kernel_size);

	// �޸� ��ũ�γ����� ����
	cudaThreadSynchronize();
}