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

__global__
void cuda_shared_Filter_2D(uchar * pSrcImage, uchar *pDstImage,
	int SrcWidth, int SrcHeight, float *pKernel, int KWidth, int KHeight)
{
	// 저장할 shared 메모리 선언
	extern __shared__ float shared[];

	// 블록과 쓰레드 주소에 따라 현재 pixel의 index를 계산한다
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y * SrcWidth + x;
	int mSize = KWidth / 2;

	if (tx < KWidth && ty < KHeight)
	{
		// shared 메모리에 size만큼 커널을 저장한다
		shared[ty * KWidth + tx] = pKernel[ty * KWidth + tx];
	}
	__syncthreads();

	float temp = 0.f;

	// Serial과 동일하게 예외처리 진행 후 각 pixel 계산
	if (x >= KWidth / 2 && y >= KHeight / 2
		&& x < SrcWidth - KWidth / 2 && y < SrcHeight - KHeight / 2)
	{

		for (int j = -mSize; j <= mSize; j++) {
			for (int i = -mSize; i <= mSize; i++) {

				// float 형태로 계산 및 저장
				temp += (float)pSrcImage[index + i + j * SrcWidth]
					* shared[i + mSize + (j + mSize) * KHeight];
			}
		}
		// 최종 dst 이미지에는 uchar의 형태로 저장
		pDstImage[index] = (uchar)temp;
	}
	else {
		pDstImage[index] = 0;		// kernel size 바깥의 픽셀의 예외처리 진행
	}
}

// 커널 사이즈만큼 미리 constant 메모리 선언
__constant__ float constKernel[5 * 5];

__global__
void cuda_constant_Filter_2D(uchar * pSrcImage, uchar *pDstImage,
	int SrcWidth, int SrcHeight, int KWidth, int KHeight)
{
	// 블록과 쓰레드 주소에 따라 현재 pixel의 index를 계산한다
	int tx = threadIdx.x;
	int ty = threadIdx.y;

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
				// constant kernel과 곱하여 계산
				temp += (float)pSrcImage[index + i + j * SrcWidth]
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
	dim3 grid = dim3(w / 16, h / 16);
	dim3 block = dim3(16, 16);

	// 각 pixel 별 CUDA Gaussain Filter 진행

	int c = 1;			// 0 : global / 1 : shared / 2 : constant

	if (c == 0)
		cuda_Filter_2D << < grid, block >> > (pcuSrc, pcuDst, w, h, cuGkernel, kernel_size, kernel_size);
	else if (c == 1)
		// 커널만큼의 size를 shared memory에 동적으로 할당한다
		cuda_shared_Filter_2D << < grid, block, sizeof(float) * 5 * 5 >> > (pcuSrc, pcuDst, w, h, cuGkernel, kernel_size, kernel_size);
	else if (c == 2) {
		cudaMemcpyToSymbol(constKernel, cuGkernel, sizeof(float)*kernel_size*kernel_size);
		cuda_constant_Filter_2D << < grid, block >> > (pcuSrc, pcuDst, w, h, kernel_size, kernel_size);
	}

	// 메모리 싱크로나이즈 진행
	cudaThreadSynchronize();
}