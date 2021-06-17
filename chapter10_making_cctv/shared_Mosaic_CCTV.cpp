
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <omp.h>
#include <ipp.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <vector>

using namespace std;
using namespace cv;

#define num 100

#define resultWidth 1024
#define resultHeight 1024

#define localWidth (resultWidth / (sqrt(num)))
#define localHeight (resultHeight / (sqrt(num)))

// 시간 계산
float fpssum = 0.f;
int iteration = 0;
int finish = 0;

// cu파일과의 연동을 위해 main code에 extern "C"로 함수 선언
extern "C" void CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst, int w, int h, float *cuGkernel, int kernel_size);
extern "C" void shared_CUDA_Gaussian_Filter(uchar *pcuSrc, uchar *pcuDst, int w, int h, float *cuGkernel, int kernel_size);

// 비디오 불러오기
void getFrame(Mat *frame, VideoCapture *cap) {
#pragma omp parallel for
	for (int i = 0; i < num; i++) {
		cap[i] >> frame[i];
		if (frame[i].empty()) {
			finish = 1;
			break;
		}
	}
}

void ippGaussianBlur(Mat &src, Mat&dst, int w, int h, int kernel_size, int sigma) {

	IppiSize tsize;
	// get the size of ROI
	tsize.width = w * 3;
	tsize.height = h - 3;
	int srcStep = 0, dstStep = 0;

	Ipp8u *S_img = (Ipp8u *)ippiMalloc_8u_C3(tsize.width, tsize.height, &srcStep);	// src IPP data
	Ipp8u *T = (Ipp8u *)ippiMalloc_8u_C3(tsize.width, tsize.height, &srcStep);		// dst IPP data

	ippiCopy_8u_C3R((const Ipp8u*)src.data, tsize.width, S_img, tsize.width, tsize);

	int iTmpBufSize = 0, iSpecSize = 0;		// Buffer size, iSpec size
	ippiFilterGaussianGetBufferSize(		// get buffer, iSpec size in kernel and ROI size
		tsize,					// ROI size
		kernel_size,			// kernel size
		ipp8u,					// IPP data type
		3,						// number of channel
		&iSpecSize, &iTmpBufSize);

	IppFilterGaussianSpec* pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
	Ipp8u* pBuffer = ippsMalloc_8u(iTmpBufSize);
	Ipp8u* border = new Ipp8u[3];
	border[0] = 255;
	border[1] = 255;
	border[2] = 255;

	// start Gaussian Blur and get blurred IPP data T
	ippiFilterGaussianInit(tsize, kernel_size, sigma, ippBorderConst, ipp8u, 3, pSpec, pBuffer);
	ippiFilterGaussianBorder_8u_C3R(S_img, tsize.width, T, tsize.width, tsize, border, pSpec, pBuffer);

	ippiCopy_8u_C3R(T, tsize.width, (Ipp8u*)src.data, tsize.width, tsize);

	// release
	ippiFree(pBuffer);
	ippiFree(pSpec);
	ippiFree(S_img);
	ippiFree(T);
}

void ompDownsampling(Mat &src, Mat&dst, int width, int height) {
	int hindex;
	int windex;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {

			hindex = j * src.rows / height;
			windex = i * src.cols / width;

			for (int k = 0; k < 3; k++) {
				dst.at<Vec3b>(j, i)[k] = src.at<Vec3b>(hindex, windex)[k];
			}
		}
	}
}

void ippGaussianDownsampling(Mat &src, Mat&dst, int ksize, int width, int height) {

	ippGaussianBlur(src, src, src.cols, src.rows, ksize, 1);
	ompDownsampling(src, dst, width, height);
}

void Gaussian_Kernel_2D(int kernel_size, float sigma, float *kernel) {
	int mSize = kernel_size / 2;
	float sum = 0.0f;

	// kernel_size * kernel_size에 알맞은 값을 저장한다
	for (int y = -mSize; y <= mSize; y++) {
		for (int x = -mSize; x <= mSize; x++) {
			kernel[x + mSize + (y + mSize) * kernel_size] =
				exp(-(pow(x, 2) + pow(y, 2)) / (2 * sigma * sigma)) /
				(2 * (float)CV_PI * sigma * sigma);
			sum += kernel[x + mSize + (y + mSize) * kernel_size];
		}
	}

	// kernel 총 덧셈 시 1이 되도록 normalization 진행
	for (int i = 0; i < kernel_size * kernel_size; i++) {
		kernel[i] /= sum;
	}
}

void cudaGaussian(Mat &src, Mat &dst, int w, int h, float *kernel, int kernel_size) {
	uchar *pcuSrc;
	uchar *pcuDst;
	//float *pcuGkernel;

	(cudaMalloc((void**)&pcuSrc, (w * h * 3) * sizeof(uchar)));
	(cudaMalloc((void**)&pcuDst, (w * h * 3) * sizeof(uchar)));

	//(cudaMalloc((void**)&pcuGkernel, kernel_size * kernel_size * sizeof(float)));

	// kernel과 src 데이터 복사
	(cudaMemcpy(pcuSrc, src.data, w * h * 3 * sizeof(uchar), cudaMemcpyHostToDevice));
	//(cudaMemcpy(pcuGkernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	// 쿠다 GPU Filter code
	shared_CUDA_Gaussian_Filter(pcuSrc, pcuDst, w, h, kernel, kernel_size);

	// 출력 영상에 GPU data를 저장
	(cudaMemcpy(dst.data, pcuDst, w * h * 3 * sizeof(uchar), cudaMemcpyDeviceToHost));

	// free the device memory
	cudaFree(pcuSrc);
	cudaFree(pcuDst);
	//cudaFree(pcuGkernel);
}

void cudaGaussianDownsampling(Mat &src, Mat&dst, int width, int height, int ksize, float *kernel) {

	cudaGaussian(src, src, src.cols, src.rows, kernel, ksize);
	ompDownsampling(src, dst, width, height);
}

void catImage(Mat &src, Mat &result, int i, int j) {
	// frame의 모든 data를 result의 정확한 위치에 복사
	for (int y = 0; y < localHeight; y++) {
		for (int x = 0; x < localWidth; x++) {

			for (int k = 0; k < 3; k++) {
				// result의 특정 위치에 현재 data 복사 
				result.at<Vec3b>(j * localHeight + y, i * localWidth + x)[k]
					= src.at<Vec3b>(y, x)[k];
			}

		}
	}
}

void makeMosaic(Mat *frame, Mat result) {
	// 모든 frame을 하나의 result로 병합한다

#pragma omp parallel for
	for (int i = 0; i < num; i++) {
	// 4 x 4의 정확한 위치에 frame 복사
		catImage(frame[i], result, i % (int)sqrt(num), i / (int)sqrt(num));
	}
}

void Downsampling(Mat *frame, int *ksize, float *kernel3, float *kernel5, float *kernel7,
	int width, int height) {
#pragma omp parallel for
	for (int i = 0; i < num; i++) {
		//ippGaussianDownsampling(frame[i], frame[i], ksize[i], width, height);

		switch (ksize[i]) {
		case 3:
			cudaGaussianDownsampling(frame[i], frame[i], width, height, ksize[i], kernel3);
			break;
		case 5:
			cudaGaussianDownsampling(frame[i], frame[i], width, height, ksize[i], kernel5);
			break;
		case 7:
			cudaGaussianDownsampling(frame[i], frame[i], width, height, ksize[i], kernel7);
			break;
		}
	}
}

int main() {

	int kernel_size = 3;
	int sigma = 1;
	float *kernel3 = new float[kernel_size * kernel_size];
	Gaussian_Kernel_2D(kernel_size, sigma, kernel3);
	float *cudakernel3;
	(cudaMemcpy(cudakernel3, kernel3, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	kernel_size = 5;
	float *kernel5 = new float[kernel_size * kernel_size];
	Gaussian_Kernel_2D(kernel_size, sigma, kernel5);
	float *cudakernel5;
	(cudaMemcpy(cudakernel5, kernel5, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	kernel_size = 7;
	float *kernel7 = new float[kernel_size * kernel_size];
	Gaussian_Kernel_2D(kernel_size, sigma, kernel7);
	float *cudakernel7;
	(cudaMemcpy(cudakernel7, kernel7, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

	string *strVideo = new string[num];
	VideoCapture *cap = new VideoCapture[num];
	Mat *frame = new Mat[num];

	Mat result = Mat(resultHeight, resultWidth, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < num; i++) {
		strVideo[i] = to_string(20) + ".mp4";
		cap[i] = VideoCapture(strVideo[i]);
	}

	// frame을 불러온다. localWidth에  따라 이미지별 kerenl size 선택
	getFrame(frame, cap);
	int krate;
	int *ksize = new int[num];
	for (int i = 0; i < num; i++) {
		krate = frame[i].cols / localWidth;
		ksize[i] = (krate % 2 == 0) ? krate + 1 : krate;
		// 3 <= kernl size <= 7
		if (ksize[i] < 3) ksize[i] = 3;
		else if (ksize[i] > 7) ksize[i] = 7;
	}

	double fstart, fend, fprocTime;
	double fps;

	// namedWindow("result", WINDOW_OPENGL);

	while (1) {
		fstart = omp_get_wtime();

		// get videos using OpenMP
		getFrame(frame, cap);
		if (finish) break;			// frame 존재하지 않을 시 종료

		// downsampling videos
		Downsampling(frame, ksize, cudakernel3, cudakernel5, cudakernel7,
			localWidth, localHeight);

		// Merge all videos
		makeMosaic(frame, result);

		// fps 체크
		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		// check fps and show text
		putText(result, "fps : " + to_string(fps),
			Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);

		cout << "interation " << iteration << " fps : " << fps << "\n";
		iteration++;	fpssum += fps;
		
		imshow("result", result);
		if (waitKey(27) == 27) {
			imwrite("result.png", result);
			break;
		}
	}

	cout << "mosiac finish!!\n";
	cout << "avg fps : " << fpssum / iteration << endl;

	cudaFree(cudakernel3);
	cudaFree(cudakernel5);
	cudaFree(cudakernel7);

	return 0;
}