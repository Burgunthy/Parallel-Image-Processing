//#include <ipp.h>
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//#define iteration 10
//
//using namespace std;
//using namespace cv;
//
//void IPP_Gaussian(Mat &src, Mat &dst, int w, int h, int kernel_size, int sigma) {
//	IppiSize size, tsize;
//	// get the size of image
//	size.width = w;
//	size.height = h;
//	// get the size of ROI
//	tsize.width = w;
//	tsize.height = h;
//
//	Ipp8u *S_img = (Ipp8u *)ippsMalloc_8u(size.width * size.height);	// src IPP data
//	Ipp8u *T = (Ipp8u *)ippsMalloc_8u(size.width * size.height);		// dst IPP data
//
//	// get the IPP data form opencv image
//	ippiCopy_8u_C1R((const Ipp8u*)src.data, size.width, S_img, size.width, size);
//
//	int iTmpBufSize = 0, iSpecSize = 0;		// Buffer size, iSpec size
//	ippiFilterGaussianGetBufferSize(		// get buffer, iSpec size in kernel and ROI size
//		tsize,					// ROI size
//		kernel_size,			// kernel size
//		ipp8u,					// IPP data type
//		1,						// number of channel
//		&iSpecSize, &iTmpBufSize);
//
//	IppFilterGaussianSpec* pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(iSpecSize);
//	Ipp8u* pBuffer = ippsMalloc_8u(iTmpBufSize);
//
//	// start Gaussian Blur and get blurred IPP data T
//	ippiFilterGaussianInit(tsize, kernel_size, sigma, ippBorderConst, ipp8u, 1, pSpec, pBuffer);
//	ippiFilterGaussianBorder_8u_C1R(S_img, size.width, T, size.width, tsize, 255, pSpec, pBuffer);
//
//	// get the opencv image form IPP data
//	dst = Mat(h, w, CV_8U, T);
//}
//
//int iiimain() {
//	// -------------------------- init
//
//	Mat pInput = imread("Knee.jpg", 0);
//	//resize(pInput, pInput, Size(1024, 1024));
//	int w = pInput.cols;
//	int h = pInput.rows;
//	Mat dstdiplay;
//
//	int kernel_size = 5;					// kernel size
//	int sigma = 3;							// sigma
//
//	// for time checking
//	vector<TickMeter> tm;
//	TickMeter tm_now;
//	float pTime;
//	float t_min = 0.0f, t_max = 0.0f;
//	float t_ave = 0.0f;
//
//	cout << "My IPP Gaussian Filter" << endl;
//
//	for (int iter = 0; iter < iteration; iter++) {
//		cout << "iteration number " << iter + 1 << " ";
//		tm.push_back(tm_now);
//		tm.at(iter).start();
//
//		// - 현재 사용 알고리즘-
//		IPP_Gaussian(pInput, dstdiplay, w, h, kernel_size, sigma);
//		// - 현재 사용 알고리즘-
//
//		tm.at(iter).stop();
//		pTime = tm.at(iter).getTimeMilli();
//		printf("processing time : %.3f ms\n", pTime);
//
//		t_ave += pTime;
//
//		if (iter == 0) {
//			t_min = pTime;
//			t_max = pTime;
//		}
//		else {
//			if (pTime < t_min) t_min = pTime;
//			if (pTime > t_max) t_max = pTime;
//		}
//	}
//	if (iteration == 1) t_ave = t_ave;
//	else if (iteration == 2) t_ave = t_ave / 2;
//	else t_ave = (t_ave - t_min - t_max) / (iteration - 2);
//
//	// print Average processing time
//	cout << endl << "Average processing time : " << (float)t_ave << " ms" << endl;
//
//	// ----------------------- end
//
//	imshow("input", pInput);
//	imshow("ipp_dst", dstdiplay);
//
//	waitKey(0);
//
//	return 0;
//}