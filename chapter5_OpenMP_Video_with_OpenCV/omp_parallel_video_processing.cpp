//#include <iostream>
//#include <string>
//
//#include <opencv2/opencv.hpp>
//#include <omp.h>
//
//using namespace cv;
//using namespace std;
//
//int DisplayVideo(string strVideo, string windowName) {
//
//	VideoCapture cap(strVideo);
//	if (!cap.isOpened()) return -1;
//
//	Mat edges;
//	namedWindow(windowName, 0);
//	double fstart, fend, fprocTime;
//	double fps;
//
//	while (1) {
//		fstart = omp_get_wtime();
//
//		Mat frame;
//		cap >> frame;
//		if (frame.empty()) {
//			destroyWindow(windowName);
//			break;
//		}
//
//		fend = omp_get_wtime();
//		fprocTime = fend - fstart;
//		fps = (1 / (fprocTime));
//		putText(frame, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);
//
//		imshow(windowName, frame);
//
//		waitKey(27);
//	}
//
//	return 0;
//}
//
//int CannyVideo(string strVideo, string windowName) {
//
//	VideoCapture cap(strVideo);
//	if (!cap.isOpened()) return -1;
//
//	Mat edges;
//	namedWindow(windowName, 0);
//	double fstart, fend, fprocTime;
//	double fps;
//
//	while (1) {
//		fstart = omp_get_wtime();
//
//		Mat frame;
//		cap >> frame;
//
//		if (frame.empty()) {
//			destroyWindow(windowName);
//			break;
//		}
//
//		cvtColor(frame, frame, COLOR_BGR2GRAY);
//		Canny(frame, frame, 100, 150);
//
//		fend = omp_get_wtime();
//		fprocTime = fend - fstart;
//		fps = (1 / (fprocTime));
//		putText(frame, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);
//
//		imshow(windowName, frame);
//
//		waitKey(27);
//	}
//
//	return 0;
//}
//
//int GaussianVideo(string strVideo, string windowName) {
//
//	VideoCapture cap(strVideo);
//	if (!cap.isOpened()) return -1;
//
//	Mat edges;
//	namedWindow(windowName, 0);
//	double fstart, fend, fprocTime;
//	double fps;
//
//	while (1) {
//		fstart = omp_get_wtime();
//
//		Mat frame;
//		cap >> frame;
//
//		if (frame.empty()) {
//			destroyWindow(windowName);
//			break;
//		}
//
//		GaussianBlur(frame, frame, Size(5, 5), 3);
//
//		fend = omp_get_wtime();
//		fprocTime = fend - fstart;
//		fps = (1 / (fprocTime));
//		putText(frame, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);
//
//		imshow(windowName, frame);
//		waitKey(27);
//	}
//
//	return 0;
//}
//
//int SobelVideo(string strVideo, string windowName) {
//
//	VideoCapture cap(strVideo);
//	if (!cap.isOpened()) return -1;
//
//	Mat edges;
//	namedWindow(windowName, 0);
//	double fstart, fend, fprocTime;
//	double fps;
//
//	Mat sobelX;
//	Mat sobelY;
//
//	while (1) {
//		fstart = omp_get_wtime();
//
//		Mat frame;
//		cap >> frame;
//
//		if (frame.empty()) {
//			destroyWindow(windowName);
//			break;
//		}
//
//		cvtColor(frame, frame, COLOR_BGR2GRAY);
//
//		Sobel(frame, sobelX, CV_8U, 1, 0);
//		Sobel(frame, sobelY, CV_8U, 0, 1);
//		frame = abs(sobelX) + abs(sobelY);
//
//		fend = omp_get_wtime();
//		fprocTime = fend - fstart;
//		fps = (1 / (fprocTime));
//		putText(frame, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);
//
//		imshow(windowName, frame);
//		waitKey(27);
//	}
//
//	return 0;
//}
//
//int ttmain() {
//
//	string strVideo = "vvv.mp4";
//
//#pragma omp parallel sections
//	{
//#pragma omp section
//	DisplayVideo(strVideo, "1");
//
//#pragma omp section
//	GaussianVideo(strVideo, "2");
//
//#pragma omp section
//	SobelVideo(strVideo, "3");
//
//#pragma omp section
//	CannyVideo(strVideo, "4");
//
//	}
//
//	waitKey(0);
//
//	return 0;
//}