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
	// ��ϰ� ������ �ּҿ� ���� ���� pixel�� index�� ����Ѵ�
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;

	int index = y * SrcWidth * 3 + x * 3 + tz;
	int mSize = KWidth / 2;

	float temp = 0.f;

	// Serial�� �����ϰ� ����ó�� ���� �� �� pixel ���
	if (x >= KWidth / 2 && y >= KHeight / 2
		&& x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{
		for (int j = -mSize; j <= mSize; j++) {
			for (int i = -mSize; i <= mSize; i++) {

				// float ���·� ��� �� ����
				temp += (float)pSrcImage[index + i * 3 + j * SrcWidth * 3]
					* constKernel[i + mSize + (j + mSize) * KHeight];
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
	dim3 grid = dim3(w / 16, h / 16, 1);
	dim3 block = dim3(16, 16, 3);

	// �� pixel �� CUDA Gaussain Filter ����

	cudaMemcpyToSymbol(constKernel, cuGkernel, sizeof(float)*kernel_size*kernel_size);
	cuda_constant_Filter_2D << < grid, block >> > (pcuSrc, pcuDst, w, h, kernel_size, kernel_size);

	// �޸� ��ũ�γ����� ����
	cudaThreadSynchronize();
}