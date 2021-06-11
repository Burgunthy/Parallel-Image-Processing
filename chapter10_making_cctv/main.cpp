#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <omp.h>
#include <ipp.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <vector>

using namespace std;
using namespace cv;

#define num 16
#define resultWidth 1024
#define resultHeight 1024

#define localWidth (resultWidth / (sqrt(num)))
#define localHeight (resultHeight / (sqrt(num)))

extern "C" void CUDA_resize(uchar* frame);

// 비디오 불러오기
void getFrame(Mat *frame, VideoCapture *cap) {
#pragma omp parallel for
	for (int i = 0; i < num; i++) {
		cap[i] >> frame[i];
		//imshow(to_string(i), frame[i]);
	}
}

void ocvResize(Mat *frame, int width, int height) {

	for (int i = 0; i < num; i++) {
		resize(frame[i], frame[i], Size(width, height));
	}
}

void ompResize(Mat *frame, int width, int height) {

}

void ippResize(Mat *frame, int width, int height) {

}

void sseResize(Mat *frame, int width, int height) {

}

void cudaResize(Mat *frame, int width, int height) {

}

void catImage(Mat &src, Mat &result, int i, int j) {
	for (int y = 0; y < localHeight; y++) {
		for (int x = 0; x < localWidth; x++) {

			for (int k = 0; k < 3; k++) {
				result.at<Vec3b>(j * localHeight + y, i * localWidth + x)[k] = src.at<Vec3b>(y, x)[k];
			}

		}
	}
}

void makeCCTV(Mat *frame, Mat result) {
	// 하나의 이미지로 변경한다

#pragma omp parallel for
	for (int i = 0; i < num; i++) {
		catImage(frame[i], result, i % (int)sqrt(num), i / (int)sqrt(num));
	}
}

int main() {

	// 비디오 불러오기 -> 16개 openmp 사용

	string *strVideo = new string[num];
	VideoCapture *cap = new VideoCapture[num];
	Mat *frame = new Mat[num];

	Mat result = Mat(resultHeight, resultWidth, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < num; i++) {
		strVideo[i] = to_string(i) + ".mp4";
		cap[i] = VideoCapture(strVideo[i]);
	}

	double fstart, fend, fprocTime;
	double fps;

	while (1) {
		fstart = omp_get_wtime();

		// 비디오 불러오기
		getFrame(frame, cap);

		// 모든 비디오 resize
		ocvResize(frame, localWidth, localHeight);

		

		// 모든 비디오 하나의 Mat으로 합침
		makeCCTV(frame, result);

		// fps 체크
		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		// check fps and show text
		putText(result, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);

		// 출력
		imshow("result", result);
		if (waitKey(27) == 27) break;
	}

	return 0;
}