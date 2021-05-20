#include <omp.h>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
void Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he);
void Filter2DMP_2(Mat src, int w, int h, Mat dst, Mat element, int we, int he);

int kSize = 3;
int mSize = kSize / 2;

int main() {
	TickMeter tm1, tm2, tm3;

	Mat element(3, 3, CV_32F);
	float FilterElm = (float)1 / (element.rows*element.cols);
	element.at<float>(0, 0) = FilterElm;	element.at<float>(0, 1) = FilterElm;	element.at<float>(0, 2) = FilterElm;
	element.at<float>(1, 0) = FilterElm;	element.at<float>(1, 1) = FilterElm;	element.at<float>(1, 2) = FilterElm;
	element.at<float>(2, 0) = FilterElm;	element.at<float>(2, 1) = FilterElm;	element.at<float>(2, 2) = FilterElm;

	cout << "<filter>" << endl;
	for (int i = 0; i < element.rows; i++) {
		for (int j = 0; j < element.cols; j++) {
			cout << element.at<float>(i, j) << "   ";
		}
		cout << endl;
	}
	cout << endl;

	Mat src = imread("hw1_2.jpg", 0);		// 512 x 512

	int width = src.cols;		int height = src.rows;
	int eWidht = element.cols;	int eHeight = element.rows;

	Mat dstCV = Mat(height, width, CV_8UC1, Scalar(0));
	Mat dstMP = Mat(height, width, CV_8UC1, Scalar(0));
	Mat dstMP2 = Mat(height, width, CV_8UC1, Scalar(0));

	cout << "<image size>" << endl;
	cout << "width : " << width << "   height : " << height << endl;
	cout << endl;

	// 시간 체크
	tm1.start();
	Filter2DCV(src, width, height, dstCV, element, eWidht, eHeight);		// 이거 만들기 dst 구하도록 ( 시리얼 코드)
	tm1.stop();
	printf("processing time : %f msec\n", tm1.getTimeMilli());

	tm2.start();
	Filter2DMP(src, width, height, dstMP, element, eWidht, eHeight);
	tm2.stop();
	printf("processing time : %f msec\n", tm2.getTimeMilli());

	tm3.start();
	Filter2DMP(src, width, height, dstMP2, element, eWidht, eHeight);
	tm3.stop();
	printf("processing time : %f msec\n", tm3.getTimeMilli());

	// 끝

	imshow("src", src);
	imshow("cv", dstCV);
	imshow("mp", dstMP);
	imshow("mp2", dstMP2);

	waitKey(0);

	return 0;
}

void Filter2DCV(Mat src, int w, int h, Mat dst, Mat element, int we, int he) {
	float sum;

	for (int i = we / 2; i < w - we / 2; i++) {
		for (int j = he / 2; j < h - he / 2; j++) {
			sum = 0.0f;

			for (int x = -we / 2; x <= we / 2; x++) {
				for (int y = -he / 2; y <= he / 2; y++) {
					sum += src.at<uchar>(i + x, j + y) * element.at<float>(x + we / 2, y + he / 2);
				}
			}
			dst.at<uchar>(i, j) = (uchar)sum;
		}
	}
}

void Filter2DMP(Mat src, int w, int h, Mat dst, Mat element, int we, int he) {
	float sum;

#pragma omp parallel for private(sum)
	for (int i = we / 2; i < w - we / 2; i++) {
		for (int j = he / 2; j < h - he / 2; j++) {
			sum = 0.0f;

			for (int x = -we / 2; x <= we / 2; x++) {
				for (int y = -he / 2; y <= he / 2; y++) {
					sum += src.at<uchar>(i + x, j + y) * element.at<float>(x + we / 2, y + he / 2);
				}
			}
			dst.at<uchar>(i, j) = (uchar)sum;
		}
	}
}

void Filter2DMP_2(Mat src, int w, int h, Mat dst, Mat element, int we, int he) {
	float sum;

#pragma omp parallel for private(sum)
	for (int i = we / 2 + (he / 2 * w); i < w * h - (we / 2 + (he / 2 * w)); i++) {
		sum = 0.0f;

		for (int x = -we / 2; x <= we / 2; x++) {
			for (int y = -he / 2; y <= he / 2; y++) {
				sum += src.data[i + x + y * w] * element.at<float>(x + we / 2, y + he / 2);
			}
		}
		dst.data[i] = (uchar)sum;
	}
}