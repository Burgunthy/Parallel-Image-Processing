#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <ipp.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define iteration 10

using namespace std;
using namespace cv;

// cu파일과의 연동을 위해 main code에 extern "C"로 함수 선언
extern "C" void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

void Gaussian_Kernel_2D(int kernel_size, float sigma, float *kernel);

void Serial_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel);
void OpenMP_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel);
void IPP_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, int sigma);
void CUDA_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel);

int ssemain() {
	// src, dst 이미지 선언
	Mat pInput = imread("Knee.jpg", 0);
	//resize(pInput, pInput, Size(1024, 1024));
	int w = pInput.cols;
	int h = pInput.rows;
	Mat dstdiplay = Mat(h, w, CV_8UC1);

	// kernel 설정 및 serial커널 계산
	int kernel_size = 5;
	float sigma = 3;
	float *Gkernel = new float[kernel_size * kernel_size];
	Gaussian_Kernel_2D(kernel_size, sigma, Gkernel);

	// for time checking
	vector<TickMeter> tm;
	TickMeter tm_now;
	float pTime;
	float t_min = 0.0f, t_max = 0.0f;
	float t_ave = 0.0f;

	cout << "My Serial Gaussian Filter" << endl;

	// 10 iteration 진행
	for (int iter = 0; iter < iteration; iter++) {
		cout << "iteration number " << iter + 1 << " ";
		tm.push_back(tm_now);
		tm.at(iter).start();

		// - 현재 사용 알고리즘-

		// Serial_Gaussian(pInput, dstdiplay, w, h, kernel_size, Gkernel);
		// OpenMP_Gaussian(pInput, dstdiplay, w, h, kernel_size, Gkernel);
		// GaussianBlur(pInput, dstdiplay, Size(kernel_size, kernel_size), sigma);
		// IPP_Gaussian(pInput, dstdiplay, w, h, kernel_size, sigma);
		CUDA_Gaussian(pInput, dstdiplay, w, h, kernel_size, Gkernel);

		// - 현재 사용 알고리즘-

		tm.at(iter).stop();
		pTime = tm.at(iter).getTimeMilli();
		printf("processing time : %.3f ms\n", pTime);

		t_ave += pTime;

		if (iter == 0) {
			t_min = pTime;
			t_max = pTime;
		}
		else {
			if (pTime < t_min) t_min = pTime;
			if (pTime > t_max) t_max = pTime;
		}
	}

	if (iteration == 1) t_ave = t_ave;
	else if (iteration == 2) t_ave = t_ave / 2;
	else t_ave = (t_ave - t_min - t_max) / (iteration - 2);

	// print Average processing time
	cout << endl << "Average processing time : " << (float)t_ave << " ms" << endl;

	// 결과 출력 및 확인
	imshow("input", pInput);
	imshow("output", dstdiplay);

	waitKey(0);

	return 0;
}

void Gaussian_Kernel_2D(int kernel_size, float sigma, float *kernel) {
	int mSize = kernel_size / 2;
	float sum = 0.0f;

	// kernel_size * kernel_size에 알맞은 값을 저장한다
	for (int y = -mSize; y <= mSize; y++) {
		for (int x = -mSize; x <= mSize; x++) {
			kernel[x + mSize + (y + mSize) * kernel_size] =
				exp(-(pow(x, 2) + pow(y, 2)) / (2 * sigma * sigma)) / (2 * (float)CV_PI * sigma * sigma);
			sum += kernel[x + mSize + (y + mSize) * kernel_size];
		}
	}

	// kernel 총 덧셈 시 1이 되도록 normalization 진행
	for (int i = 0; i < kernel_size * kernel_size; i++) {
		kernel[i] /= sum;
	}
}

void Serial_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel) {
	int mSize = kernel_size / 2;
	float temp;

	for (int index = 0; index < w * h; index++) {
		temp = 0.f;

		// kernel_size에 해당하지 않는 boundary 부분 예외 처리
		if ((index % w >= mSize) && (index / w >= mSize) && (index % w < w - mSize) && (index / w < h - mSize)) {
			for (int j = -mSize; j <= mSize; j++) {
				for (int i = -mSize; i <= mSize; i++) {
					// float으로 계산한다
					temp += (float)src.data[index + i + j * w] * kernel[i + mSize + (j + mSize) * kernel_size];
				}
			}
			// uchar의 형태로 저장한다
			dst.data[index] = (uchar)temp;
		}
		// boundary는 모두 0으로 처리한다
		else dst.data[index] = 0;
	}
}

void OpenMP_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel) {
	int mSize = kernel_size / 2;
	float temp;

	// OpenMP 과정 추가
#pragma omp parallel for
	for (int index = 0; index < w * h; index++) {
		temp = 0.f;

		// kernel_size에 해당하지 않는 boundary 부분 예외 처리
		if ((index % w >= mSize) && (index / w >= mSize) && (index % w < w - mSize) && (index / w < h - mSize))
		{
			for (int j = -mSize; j <= mSize; j++) {
				for (int i = -mSize; i <= mSize; i++) {

					// float으로 계산한다
					temp += (float)src.data[index + i + j * w] * kernel[i + mSize + (j + mSize) * kernel_size];
				}
			}
			// uchar의 형태로 저장한다
			dst.data[index] = (uchar)temp;
		}
		// boundary는 모두 0으로 처리한다
		else dst.data[index] = 0;
	}
}

void IPP_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, int sigma) {
	IppiSize size, tsize;
	// get the size of image
	size.width = w;
	size.height = h;
	// get the size of ROI
	tsize.width = w;
	tsize.height = h;

	Ipp8u *S_img = (Ipp8u *)ippsMalloc_8u(size.width * size.height);	// src IPP data
	Ipp8u *T = (Ipp8u *)ippsMalloc_8u(size.width * size.height);		// dst IPP data

	// get the IPP data form opencv image
	ippiCopy_8u_C1R((const Ipp8u*)src.data, size.width, S_img, size.width, size);

	int iTmpBufSize = 0, iSpecSize = 0;		// Buffer size, iSpec size
	ippiFilterGaussianGetBufferSize(		// get buffer, iSpec size in kernel and ROI size
		tsize,					// ROI size
		kernel_size,			// kernel size
		ipp8u,					// IPP data type
		1,						// number of channel
		&iSpecSize, &iTmpBufSize);

	IppFilterGaussianSpec* pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	Ipp8u* pBuffer = ippsMalloc_8u(iTmpBufSize);

	// start Gaussian Blur and get blurred IPP data T
	ippiFilterGaussianInit(tsize, kernel_size, sigma, ippBorderConst, ipp8u, 1, pSpec, pBuffer);
	ippiFilterGaussianBorder_8u_C1R(S_img, size.width, T, size.width, tsize, 255, pSpec, pBuffer);

	// get the opencv image form IPP data
	dst = Mat(h, w, CV_8U, T);
}

void CUDA_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel) {
	uchar *pcuSrc;
	uchar *pcuDst;
	float *pcuGkernel;

	// GPU 메모리 할당. 계산 시 float이 이용되므로 kernel만 float으로 선언
	(cudaMalloc((void**)&pcuSrc, w*h * sizeof(uchar)));
	(cudaMalloc((void**)&pcuDst, w*h * sizeof(uchar)));
	(cudaMalloc((void**)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));

	// kernel과 src 데이터 복사
	(cudaMemcpy(pcuSrc, src.data, w*h * sizeof(uchar), cudaMemcpyHostToDevice));
	(cudaMemcpy(pcuGkernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	// 쿠다 GPU Filter code
	CUDA_Gaussian_Filter(pcuSrc, pcuDst, w, h, pcuGkernel, kernel_size);

	// 출력 영상에 GPU data를 저장
	(cudaMemcpy(dst.data, pcuDst, w * h * sizeof(uchar), cudaMemcpyDeviceToHost));

	// free the device memory
	cudaFree(pcuSrc);
	cudaFree(pcuDst);
	cudaFree(pcuGkernel);
}