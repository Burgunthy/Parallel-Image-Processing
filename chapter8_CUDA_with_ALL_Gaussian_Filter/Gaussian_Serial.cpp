#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#define iteration 10

using namespace std;
using namespace cv;

void Serial_Gaussian_Kernel_2D(int kernel_size, float sigma, float *kernel) {
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

int main() {
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
	Serial_Gaussian_Kernel_2D(kernel_size, sigma, Gkernel);

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
		Serial_Gaussian(pInput, dstdiplay, w, h, kernel_size, Gkernel);
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
