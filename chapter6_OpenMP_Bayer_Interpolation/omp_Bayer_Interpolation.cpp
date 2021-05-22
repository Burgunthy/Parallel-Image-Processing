#define _CRT_SECURE_NO_WARNINGS

// #include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

#define iteration 10

// 8 bit -> 10 bit
void seq_data_copy(unsigned char* buffer, unsigned short* data, int size)
{
	//unsigned short temp;
	for (int i = 0, j = 0; i < size; i += 4, j += 5)
	{
		// for 10 bit
		data[i] = (buffer[j] << 2) + ((buffer[j + 4] >> 0) & 3);
		data[i + 1] = (buffer[j + 1] << 2) + ((buffer[j + 4] >> 2) & 3);
		data[i + 2] = (buffer[j + 2] << 2) + ((buffer[j + 4] >> 4) & 3);
		data[i + 3] = (buffer[j + 3] << 2) + ((buffer[j + 4] >> 6) & 3);
	}
}

// check all 4 corner in image
void checkCorner(Mat &RGB, unsigned short * data,
	int w, int h, int st_row, int end_row, int st_col, int end_col) {

	// Red
	RGB.at<Vec3w>(st_row, st_col)[2] = data[st_row * w + st_col];
	RGB.at<Vec3w>(st_row, end_col)[2] = data[(st_row * w + end_col) - 1];
	RGB.at<Vec3w>(end_row, st_col)[2] = data[(end_row * w + st_col) - w];
	RGB.at<Vec3w>(end_row, end_col)[2] = data[(end_row * w + end_col) - w - 1];
	// Green
	RGB.at<Vec3w>(st_row, st_col)[1] =
		(data[(st_row * w + st_col) + 1] + data[(st_row * w + st_col) + w]) / 2.0;
	RGB.at<Vec3w>(st_row, end_col)[1] = data[st_row * w + end_col];
	RGB.at<Vec3w>(end_row, st_col)[1] = data[end_row * w + st_col];
	RGB.at<Vec3w>(end_row, end_col)[1] =
		(data[(end_row * w + end_col) - 1] + data[(end_row * w + end_col) - w]) / 2.0;
	// Blue
	RGB.at<Vec3w>(st_row, st_col)[0] = data[(st_row * w + st_col) + 1 + w];
	RGB.at<Vec3w>(st_row, end_col)[0] = data[(st_row * w + end_col) + w];
	RGB.at<Vec3w>(end_row, st_col)[0] = data[(end_row * w + st_col) + 1];
	RGB.at<Vec3w>(end_row, end_col)[0] = data[(end_row * w + end_col)];
}

void checkRow(Mat &RGB, unsigned short * data,
	int w, int h, int st_row, int end_row, int st_col, int end_col) {
	int index1, index2;
	int check1 = st_row * w;				// start point in row
	int check2 = end_row * w;				// last point in row

	for (int i = st_col + 1; i < end_col; i++) {
		index1 = check1 + i;				// first row index
		index2 = check2 + i;				// last row index

		if (i % 2 == 0) {					// even column
			// firstRow		(R, G, B)
			RGB.at<Vec3w>(st_row, i)[2] = data[index1];
			RGB.at<Vec3w>(st_row, i)[1] = (data[index1 - 1] + data[index1 + 1] + data[index1 + w]) / 3.0;
			RGB.at<Vec3w>(st_row, i)[0] = (data[index1 + w - 1] + data[index1 + w + 1]) / 2.0;

			// lastRow		(R, G, B)
			RGB.at<Vec3w>(end_row, i)[2] = data[index2 - w];
			RGB.at<Vec3w>(end_row, i)[1] = data[index2];
			RGB.at<Vec3w>(end_row, i)[0] = (data[index2 - 1] + data[index2 + 1]) / 2.0;
		}
		else {								// odd column
			// firstRow		(R, G, B)
			RGB.at<Vec3w>(st_row, i)[2] = (data[index1 - 1] + data[index1 + 1]) / 2.0;
			RGB.at<Vec3w>(st_row, i)[1] = data[index1];
			RGB.at<Vec3w>(st_row, i)[0] = data[index1 + w];

			// lastRow		(R, G, B)
			RGB.at<Vec3w>(end_row, i)[2] = (data[index2 - w - 1] + data[index2 - w + 1]) / 2.0;
			RGB.at<Vec3w>(end_row, i)[1] = (data[index2 - 1] + data[index2 + 1] + data[index2 - w]) / 3.0;
			RGB.at<Vec3w>(end_row, i)[0] = data[index2];
		}
	}
}

void checkCol(Mat &RGB, unsigned short * data,
	int w, int h, int st_row, int end_row, int st_col, int end_col) {
	int index1, index2;
	int check1 = st_col;					// start point in column
	int check2 = end_col;					// start point in column

	for (int j = st_row + 1; j < end_row; j++) {
		index1 = check1 + j * w;
		index2 = check2 + j * w;

		if (j % 2 == 0) {					// even row
			// firstCol		(R, G, B)
			RGB.at<Vec3w>(j, st_col)[2] = data[index1];
			RGB.at<Vec3w>(j, st_col)[1] = (data[index1 - w] + data[index1 + w] + data[index1 + 1]) / 3.0;
			RGB.at<Vec3w>(j, st_col)[0] = (data[index1 - w + 1] + data[index1 + w + 1]) / 2.0;

			// lastCol		(R, G, B)
			RGB.at<Vec3w>(j, end_col)[2] = data[index2 - 1];
			RGB.at<Vec3w>(j, end_col)[1] = data[index2];
			RGB.at<Vec3w>(j, end_col)[0] = (data[index2 - w] + data[index2 + w]) / 2.0;
		}
		else {								// odd row
			// firstCol		(R, G, B)
			RGB.at<Vec3w>(j, st_col)[2] = (data[index1 - w] + data[index1 + w]) / 2.0;
			RGB.at<Vec3w>(j, st_col)[1] = data[index1];
			RGB.at<Vec3w>(j, st_col)[0] = data[index1 + 1];

			// lastCol		(R, G, B)
			RGB.at<Vec3w>(j, end_col)[2] = (data[index2 - w - 1] + data[index2 + w - 1]) / 2.0;
			RGB.at<Vec3w>(j, end_col)[1] = (data[index2 - w] + data[index2 + w] + data[index2 - 1]) / 3.0;
			RGB.at<Vec3w>(j, end_col)[0] = data[index2];
		}
	}
}

// 무조건 4배수로!!!!

void interR(Mat &RGB, unsigned short * data, int w, int h, int i, int j) {
	int index = i + j * w;		// current pixel address

	if (i % 2 == 0) {			// even column
		if (j % 2 == 0) {		// even column, even row
			RGB.at<Vec3w>(j, i)[2] = data[index];
		}
		else {					// even col, odd row
			RGB.at<Vec3w>(j, i)[2] = (data[index - w] + data[index + w]) / 2;
		}
	}
	else {						// odd column
		if (j % 2 == 0) {		// odd column, even row
			RGB.at<Vec3w>(j, i)[2] = (data[index - 1] + data[index + 1]) / 2;
		}
		else {					// odd column, odd row
			RGB.at<Vec3w>(j, i)[2] =
				(data[index - w - 1] + data[index - w + 1] +
					data[index + w - 1] + data[index + w + 1]) / 4;
		}
	}
}

void interG(Mat &RGB, unsigned short * data, int w, int h, int i, int j) {
	int index = i + j * w;		// current pixel address

	if (i % 2 == 0) {			// even column
		if (j % 2 == 0) {		// even column, even row
			RGB.at<Vec3w>(j, i)[1] =
				(data[index - 1] + data[index + 1] +
					data[index - w] + data[index + w]) / 4;
		}
		else {					// even col, odd row
			RGB.at<Vec3w>(j, i)[1] = data[index];
		}
	}
	else {						// odd column
		if (j % 2 == 0) {		// odd column, even row
			RGB.at<Vec3w>(j, i)[1] = data[index];
		}
		else {					// odd column, even row
			RGB.at<Vec3w>(j, i)[1] =
				(data[index - 1] + data[index + 1] +
					data[index - w] + data[index + w]) / 4;
		}
	}
}

void interB(Mat &RGB, unsigned short * data, int w, int h, int i, int j) {
	int index = i + j * w;		// current pixel address

	if (i % 2 == 0) {			// even column
		if (j % 2 == 0) {		// even column, even row
			RGB.at<Vec3w>(j, i)[0] =
				(data[index - 1] + data[index + 1] +
					data[index - w] + data[index + w]) / 4;
		}
		else {					// even col, odd row
			RGB.at<Vec3w>(j, i)[0] = (data[index - 1] + data[index + 1]) / 2;
		}
	}
	else {						// odd column
		if (j % 2 == 0) {		// odd column, even row
			RGB.at<Vec3w>(j, i)[0] = (data[index - w] + data[index + w]) / 2;
		}
		else {					// odd column, even row
			RGB.at<Vec3w>(j, i)[0] = data[index];
		}
	}
}

inline void Interp_rggb_ushort_seq(Mat &RGB, unsigned short * data,
	int w, int h,
	int st_row, int end_row, int st_col, int end_col) {

	// calculate four corner
	checkCorner(RGB, data, w, h, st_row, end_row, st_col, end_col);
	// calculate row edge
	checkRow(RGB, data, w, h, st_row, end_row, st_col, end_col);
	// calculate column edge
	checkCol(RGB, data, w, h, st_row, end_row, st_col, end_col);

	// calculate inside pixel
	for (int j = st_row + 1; j < end_row; j++) {
		for (int i = st_col + 1; i < end_col; i++) {
			// R
			interR(RGB, data, w, h, i, j);
			// G
			interG(RGB, data, w, h, i, j);
			// B
			interB(RGB, data, w, h, i, j);
		}
	}
}

inline void Interp_rggb_ushort_seq_inner(Mat &RGB, unsigned short * data,
	int w, int h,
	int st_row, int end_row, int st_col, int end_col) {

	// calculate column edge
	checkCol(RGB, data, w, h, st_row - 1, end_row + 1, st_col, end_col);

	// calculate inside pixel
	for (int j = st_row; j <= end_row; j++) {
		for (int i = st_col + 1; i < end_col; i++) {
			// R
			interR(RGB, data, w, h, i, j);
			// G
			interG(RGB, data, w, h, i, j);
			// B
			interB(RGB, data, w, h, i, j);
		}
	}
}

int main() {
	omp_set_num_threads(4);

	int w = 3264; // image width
	int h = 2448; // image height
	//******************* File Read ***********************//

	FILE *pFile; // File pointer
	long lSize;
	unsigned char* raw;
	size_t result;

	pFile = fopen("raw.raw", "rb");
	if (pFile == NULL) { fputs("File error", stderr); exit(1); }

	// obtain file size:
	fseek(pFile, 0, SEEK_END);
	lSize = ftell(pFile);
	rewind(pFile);

	// allocate memory to contain the whole file:
	raw = (unsigned char*)malloc(sizeof(unsigned char)*lSize);
	if (raw == NULL) { fputs("Memory error", stderr); exit(2); }

	// copy the file into the buffer:
	result = fread(raw, 1, lSize, pFile);
	if (result != lSize) { fputs("Reading error", stderr); exit(3); }

	// data which for save processed raw data
	unsigned short* data = (unsigned short*)malloc(sizeof(unsigned short)*h*w);

	// 8bit data to 10 bit data( raw -> data)
	seq_data_copy(raw, data, w * h);

	Mat mRGB = Mat(h, w, CV_16UC3, Scalar(0, 0, 0));	// 10bit result image
	Mat dst;											// normalized result image

	// for time checking
	vector<TickMeter> tm;
	TickMeter tm_now;
	float pTime;
	float t_min = 0.0f, t_max = 0.0f;
	float t_ave = 0.0f;

	// Interpolation
	for (int iter = 0; iter < iteration; iter++) {

		cout << "iteration number " << iter + 1 << " ";
		tm.push_back(tm_now);
		tm.at(iter).start();

		// Serial source code
		//Interp_rggb_ushort_seq(mRGB, data, w, h, 0, h - 1, 0, w - 1);

		// First Row, Last Row를 병렬 처리 전 미리 계산한다
		// calculate four corner
		checkCorner(mRGB, data, w, h, 0, h - 1, 0, w - 1);
		// calculate row edge
		checkRow(mRGB, data, w, h, 0, h - 1, 0, w - 1);
		
		// OpenMP code using Section
		// Serial Code와는 다른 함수를 사용해 병렬처리를 진행한다
#pragma omp parallel sections
		{
#pragma omp section			// 0이 아닌 1부터 시작한다
			Interp_rggb_ushort_seq_inner(mRGB, data, w, h, 1, (h / 4 - 1), 0, w - 1);
#pragma omp section			// second
			Interp_rggb_ushort_seq_inner(mRGB, data, w, h, h / 4, (h / 2 - 1), 0, w - 1);
#pragma omp section			// third
			Interp_rggb_ushort_seq_inner(mRGB, data, w, h, h / 2, (3 * h / 4 - 1), 0, w - 1);
#pragma omp section			// h - 1이 아닌 h - 2에서 종료한다
			Interp_rggb_ushort_seq_inner(mRGB, data, w, h, 3 * h / 4, h - 2, 0, w - 1);
		}
		
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

	// normalize 10bit image to 8bit image
	normalize(mRGB, dst, 0, 255, NORM_MINMAX, CV_8UC3);

	cout << dst.cols << " " << dst.rows << endl;

	imwrite("dst.png", dst);
	waitKey(0);

	fclose(pFile);
	free(raw);

	cout << "finish" << endl;

	return 0;
}