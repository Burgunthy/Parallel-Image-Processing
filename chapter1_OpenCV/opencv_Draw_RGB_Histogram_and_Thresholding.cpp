#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat img = imread("HW1_images/hw1_2.jpg", 1);

	// record the frequency of each pixel per channel
	int histogramB[256];
	int histogramG[256];
	int histogramR[256];		

	// Initialize the value
	for (int i = 0; i < 256; i++) {
		histogramB[i] = 0;
		histogramG[i] = 0;
		histogramR[i] = 0;		
	}

	// increce the histogram array matching each pixel by 1
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histogramB[((int)img.at<uchar>(i, j, 0))]++;
			histogramG[((int)img.at<uchar>(i, j, 1))]++;
			histogramR[((int)img.at<uchar>(i, j, 2))]++;
		}
	}

	// record the frequency and value of the intensity with the maximum value.
	int max_B = 0;
	int max_G = 0;
	int max_R = 0;
	int thresh_B, thresh_G, thresh_R;
		
	for (int i = 0; i < 256; i++) {
		if (max_B < histogramB[i]) {
			max_B = histogramB[i];
			thresh_B = i;
		}
		if (max_G < histogramG[i]) {
			max_G = histogramG[i];
			thresh_G = i;
		}
		if (max_R < histogramR[i]) {
			max_R = histogramR[i];
			thresh_R = i;
		}
	}

	// draw a histogram based on opencv
	Mat histImageB(256, 256, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageG(256, 256, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageR(256, 256, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < 256; i++) {
		line(histImageB, Point(i, 256),
			Point(i, 256 - int( histogramB[i] * 256 / max_B)),	// since we know the maximum frequency
																//normalize each pixel like this.
			Scalar(255, 0, 0), 2, 8, 0);

		line(histImageG, Point(i, 256),
			Point(i, 256 - int(histogramG[i] * 256 / max_G)),
			Scalar(0, 255, 0), 2, 8, 0);

		line(histImageR, Point(i, 256),
			Point(i, 256 - int(histogramR[i] * 256 / max_R)),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	
	imshow("img", img);
	imshow("HistogramB", histImageB);
	imshow("HistogramG", histImageG);
	imshow("HistogramR", histImageR);

	waitKey(0);

	// OTSU can only receive gray images as input
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat img_threshold;
	Mat	img_auto;

	// set the optimal threshold value, then binaries the image
	// thresh is most frequent intensity in each channel

	// manually
	threshold(gray, img_threshold, thresh_R, 255, THRESH_BINARY);
	imshow("threshold_B", img_threshold);

	imwrite("last.png", img_threshold);

	waitKey(0);

	// automatically ( valu thresh does nothing in OTSU)
	threshold(gray, img_auto, thresh_B, 255, THRESH_OTSU);

	imshow("auto", img_auto);
	imwrite("auto.png", img_auto);
	waitKey(0);

	return 0;
}