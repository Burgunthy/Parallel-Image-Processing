#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <omp.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define iteration 10

using namespace std;
using namespace cv;

// cu파일과의 연동을 위해 main code에 extern "C"로 함수 선언
extern "C" void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

void Gaussian_Kernel_2D(int kernel_size, float sigma, float *kernel);

void CUDA_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel);
void CUDA_multi_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel);

int main() {
	// src, dst 이미지 선언
	Mat pInput = imread("Grab_Image.bmp", 0);
	resize(pInput, pInput, Size(6024, 6024));
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
		//GaussianBlur(pInput, dstdiplay, Size(5, 5), 3);
		CUDA_Gaussian(pInput, dstdiplay, w, h, kernel_size, Gkernel);
		//CUDA_multi_Gaussian(pInput, dstdiplay, w, h, kernel_size, Gkernel);

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

	imwrite("src.png", pInput);
	imwrite("dst.png", dstdiplay);

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

void CUDA_multi_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, float *kernel) {
	uchar *pcuSrc;
	uchar *pcuDst;
	float *pcuGkernel;

	int dev = 0;
	cudaError_t error_id = cudaGetDeviceCount(&dev);

	cout << "\nmy dev : " << dev << "\n";

	for (int i = 0; i < dev; i++) {
		cudaSetDevice(i);
		cudaMalloc((void**)&pcuSrc, w * h / dev * sizeof(float));
		cudaMalloc((void**)&pcuDst, w * h / dev * sizeof(float));
		cudaMalloc((void**)&pcuGkernel, kernel_size*kernel_size * sizeof(float));

		(cudaMemcpy(pcuGkernel, kernel, kernel_size * kernel_size * sizeof(float), 
			cudaMemcpyHostToDevice));
	}

	// 쿠다 GPU Filter code
#pragma omp parallel sections
	{
#pragma omp section
		{
			cudaSetDevice(0);
			// 이미지 불러오기 (첫 파트)
			(cudaMemcpy(pcuSrc, src.data, w * (h / 2) * sizeof(uchar), cudaMemcpyHostToDevice));
			// 가우시안 계산
			CUDA_Gaussian_Filter(pcuSrc, pcuDst, w, (h / 2), pcuGkernel, kernel_size);
			printf("A: %d _ %d \n", omp_in_parallel(), omp_get_thread_num());
			// 이미지 저장 (첫 파트)
			(cudaMemcpy(dst.data, pcuDst, w * (h / 2) * sizeof(uchar), cudaMemcpyDeviceToHost));

			// free the device memory
			cudaFree(pcuSrc);
			cudaFree(pcuDst);
			cudaFree(pcuGkernel);
		}
#pragma omp section
		{
			cudaSetDevice(1);
			// 이미지 불러오기 (두번째 파트)
			(cudaMemcpy(pcuSrc, src.data + (w*(h/2)+1), w*(h/2) * sizeof(uchar), cudaMemcpyHostToDevice));
			// 가우시안 계산
			CUDA_Gaussian_Filter(pcuSrc, pcuDst, w, (h/2), pcuGkernel, kernel_size);
			printf("B: %d _ %d \n", omp_in_parallel(), omp_get_thread_num());
			// 이미지 저장 (두번째 파트)
			(cudaMemcpy(dst.data + (w*(h/2)+1), pcuDst, w*(h/2) * sizeof(uchar), cudaMemcpyDeviceToHost));

			// free the device memory
			cudaFree(pcuSrc);
			cudaFree(pcuDst);
			cudaFree(pcuGkernel);
		}
	}

	// 출력 영상에 GPU data를 저장
}