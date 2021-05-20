#include <ipp.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main_t () {
	// -------------------------- init

	TickMeter tm1, tm2;

	Mat image = imread("hw1_2.jpg", 0);
	if (image.empty()) return 0;

	// resize image for HW2-3
	// resize(image, image, Size(image.cols / 2, image.rows / 2));		// 256 x 256
	//resize(image, image, Size(image.cols, image.rows));				// 512 x 512
	// resize(image, image, Size(image.cols * 2, image.rows * 2));		// 1024 x 1024
	resize(image, image, Size(image.cols * 4, image.rows * 4));		// 2048 x 2048

	int kSize = 5;					// kernel size

	// -------------------------- OpenCV
	tm1.start();

	Mat cv_dst;
	// test medianBlur blur in opencv
	medianBlur(image, cv_dst, kSize);

	tm1.stop();
	printf("\n    processing time : %f msec\n", tm1.getTimeMilli());

	// -------------------------- IPP
	tm2.start();

	IppiSize size, tsize;
	// get the size of image
	size.width = image.cols;
	size.height = image.rows;
	// get the size of ROI
	tsize.width = image.cols;
	tsize.height = image.rows;

	// get the kernel size
	IppiSize maskSize = { kSize, kSize };

	Ipp8u *S_img = (Ipp8u *)ippsMalloc_8u(size.width * size.height);	// src IPP data
	Ipp8u *T_img = (Ipp8u *)ippsMalloc_8u(size.width * size.height);	// buffer IPP data
	Ipp8u *T = (Ipp8u *)ippsMalloc_8u(size.width * size.height);		// dst IPP data

	// get the IPP data form opencv image
	ippiCopy_8u_C1R((const Ipp8u*)image.data, size.width, S_img, size.width, size);
	
	// start Gaussian Blur and get blurred IPP data T
	ippiFilterMedianBorder_8u_C1R(S_img, size.width, T, size.width, tsize, maskSize, ippBorderConst, 255, T_img);

	// get the opencv image form IPP data
	Size s;
	s.width = image.cols;
	s.height = image.rows;
	Mat ipp_dst(s, CV_8U, T);

	tm2.stop();

	printf("\n    processing time : %f msec\n", tm2.getTimeMilli());

// ----------------------- end

	imshow("cv_dst", cv_dst);
	imshow("ipp_dst", ipp_dst);

	waitKey(0);

	return 0;
}