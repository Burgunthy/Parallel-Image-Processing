#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <cmath>

#define iteration 10

using namespace cv;
using namespace std;

int cccmain() {

	Mat pInput = imread("Knee.jpg", 0);
	resize(pInput, pInput, Size(6024, 6024));
	int w = pInput.cols;
	int h = pInput.rows;
	Mat dstdiplay;

	// kernel 설정
	int kernel_size = 5;
	float sigma = 3;

	// for time checking
	vector<TickMeter> tm;
	TickMeter tm_now;
	float pTime;
	float t_min = 0.0f, t_max = 0.0f;
	float t_ave = 0.0f;

	cout << "My OpenCV Gaussian Filter" << endl;

	for (int iter = 0; iter < iteration; iter++) {
		cout << "iteration number " << iter + 1 << " ";
		tm.push_back(tm_now);
		tm.at(iter).start();

		// - 현재 사용 알고리즘 (OpenCV) -
		GaussianBlur(pInput, dstdiplay, Size(kernel_size, kernel_size), sigma);
		// - 현재 사용 알고리즘 (OpenCV) -

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

	imshow("input", pInput);
	imshow("output", dstdiplay);
	waitKey(0);

	return 0;

}