#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
	Mat img = imread("HW1_images/hw1_1.ppm", 0);		// HW 1 image
	Mat dst1, dst2, dst3;

	medianBlur(img, dst1, 3);							// kernel size = 3
	// medianBlur(img, dst2, 5);							// kernel size = 5
	// medianBlur(img, dst3, 7);							// kernel size = 7

	GaussianBlur(img, dst2, Size(3, 3), 3, 3);		// compared with GaussianBlur
	blur(img, dst3, Size(3, 3));						// compared with Blur

	//namedWindow("img", 0);					// resizing when window size change
	imshow("img", img);

	imshow("dst1", dst1);						// show blurred image
	imshow("dst2", dst2);
	imshow("dst3", dst3);

	waitKey(0);

	return 0;
}