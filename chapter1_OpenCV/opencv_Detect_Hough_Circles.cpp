#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main_1() {

	Mat img;

	img = imread("HW1_images/circle_blob1.png", 1);
	//img = imread("HW1_images/circle_blob2.png", 1);
	//img = imread("HW1_images/circle_blob3.png", 1);
	//img = imread("HW1_images/circle_blob4.png", 1);
	imwrite("original.png", img);

	Mat buff;

	cvtColor(img, buff, CV_BGR2GRAY);			// have to use Gray image in Blob Detector

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();	// Ptr of detector
	vector<KeyPoint> keypoints; detector->detect(buff, keypoints);		// search keypoints in image
	Mat img_Blob;
	drawKeypoints(img, keypoints, img_Blob, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// combine the original image with keypoints
	imshow("img", img_Blob);
	imwrite("output.png", img_Blob);

	waitKey(0);

	return 0;
}

int main_2() {

	Mat img;

	img = imread("HW1_images/circle_blob1.png", 1);
	// img = imread("HW1_images/circle_blob2.png", 1);
	// img = imread("HW1_images/circle_blob3.png", 1);
	// img = imread("HW1_images/circle_blob4.png", 1);	// have to use Gray image in HoughCircles

	imshow("original", img);
	imwrite("original.png", img);

	Mat img_hough;									// result image belonging circles
	img.copyTo(img_hough);
	cvtColor(img_hough, img_hough, COLOR_BGR2GRAY);
	medianBlur(img_hough, img_hough, 5);
	// you can use Blur. And it increase accuracy

	vector<Vec3f> circles;							// vector that store circles
	HoughCircles(img_hough, circles, HOUGH_GRADIENT, 1, 5, 200, 40, 0, 0);
	// img_houghC		:	input gray image
	// circles			:	result of HoughCircles function
	// HOUGH_GRADIENT	:	method of detecting circles
	// 1				:	image resolution ( 1 is original )
	// 100				:	min distance of detecting
	// 200				:	high threshold in Canny edge detection
	// 80				:	low threshold in Canny edge detection
	// 0				:	min_radius ( 0 is all )
	// 0				:	max_radius ( 0 is all )

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center(c[0], c[1]);
		int radius = c[2];

		circle(img, center, radius, Scalar(0, 255, 0), 2);
		// img			:	image where drawing circle ( storing in original image )
		// center		:	circle's center position in image
		// scalar(,,)	:	color
		// 2			:	thickness
		// ..
		// ..
	}

	imshow("img", img);
	imwrite("output.png", img);

	waitKey(0);

	return 0;
}