#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <cmath>
#include <iomanip>

#define iteration 10

using namespace std;
using namespace cv;

extern "C" void Gaussian_Filter_2D(float *pcuSrc, float *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

void Gaussian_Kernel_2D(int kernel_size, float sigma, float *kernel) {
	int mSize = kernel_size / 2;

	for (int y = -mSize; y <= mSize; y++) {
		for (int x = -mSize; x <= mSize; x++) {
			kernel[x + mSize + (y + mSize) * kernel_size] =
				exp(-(pow(x, 2) + pow(y, 2)) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
		}
	}
}

int gagmain() {

	Mat pInput = imread("Grab_Image.bmp", 0);
	namedWindow("input", 0);
	namedWindow("output", 0);
	imshow("input", pInput);

	int w = pInput.cols;
	int ws = pInput.cols;		// local?
	int h = pInput.rows;

	printf("%d\t%d\t%d\n", h, w, ws);

	Mat pfInput;
	Mat dstdiplay;
	float *pDst = new float[w*h];
	pInput.convertTo(pfInput, CV_32FC1);

	float *pcuSrc;
	float *pcuDst;
	float *pcuGkernel;

	// Allocate cuda device memory
	(cudaMalloc((void**)&pcuSrc, w*h * sizeof(float)));
	(cudaMalloc((void**)&pcuDst, w*h * sizeof(float)));

	// copy input image across to the device
	(cudaMemcpy(pcuSrc, pfInput.data, w*h * sizeof(float), cudaMemcpyHostToDevice));

	// kernel 설정
	int kernel_size = 5;
	float sigma = 0.5;
	float *Gkernel = new float[kernel_size * kernel_size];

	Gaussian_Kernel_2D(kernel_size, sigma, Gkernel);

	(cudaMalloc((void**)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));
	(cudaMemcpy(pcuGkernel, Gkernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++)
		{
			printf("%f\t", Gkernel[i*kernel_size + j]);
		}
		printf("\n");
	}

	printf("\n");

	// for time checking
	vector<TickMeter> tm;
	TickMeter tm_now;
	float pTime;
	float t_min = 0.0f, t_max = 0.0f;
	float t_ave = 0.0f;

	cout << "My Cuda Gaussian Filter" << endl;

	for (int iter = 0; iter < iteration; iter++) {
		cout << "iteration number " << iter + 1 << " ";
		tm.push_back(tm_now);
		tm.at(iter).start();

		// --------------------
		//(cudaMalloc((void**)&pcuSrc, w*h * sizeof(float)));
		//(cudaMalloc((void**)&pcuDst, w*h * sizeof(float)));
		//(cudaMemcpy(pcuSrc, pfInput.data, w*h * sizeof(float), cudaMemcpyHostToDevice));

		// GPU 가우시안 필터 1D 사용
		Gaussian_Filter_2D(pcuSrc, pcuDst, w, h, pcuGkernel, kernel_size);

		// Copy the marker data back to the host
		(cudaMemcpy(pDst, pcuDst, w*h * sizeof(float), cudaMemcpyDeviceToHost));

		Mat imgd1(Size(pInput.cols, pInput.rows), CV_32FC1, pDst);
		imgd1.convertTo(dstdiplay, CV_8UC1);

		// --------------------

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

	imshow("output", dstdiplay);
	waitKey(0);

	// free the device memory
	cudaFree(pcuSrc);
	cudaFree(pcuDst);

	return 0;
}