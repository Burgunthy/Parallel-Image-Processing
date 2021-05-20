#include <iostream>
#include <omp.h>
#include <time.h>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

#define _nx 2
#define _ny 1
#define iteration 10
#define num_thread 4		// the number of threads to use for the operation

void wInter(int x, int y, const char * wf, float* w);
void Interp(unsigned char * src, int h, int width,
	const char * wf, float * w, int x, int y, unsigned char * output);
void Interp_omp(unsigned char * src, int h, int width,
	const char * wf, float * w, int x, int y, unsigned char * output);

int main() {
	omp_set_num_threads(num_thread);	// threads to use for the operation
	int nx = _nx;						// the number of pixels for interpolation
	int ny = _ny;

	// source image (512 x 512)
	Mat src = imread("hw1_2.jpg", IMREAD_GRAYSCALE);

	resize(src, src, Size(1024, 1024));				// check 1024 x 1021

	int wd = src.cols;
	int hg = src.rows;
	int width = src.cols;
	int height = src.rows;

	// Alocate memory to dst image
	Mat dst = Mat(height*ny, width*nx, CV_8UC1, Scalar(0));

	// for time checking
	vector<TickMeter> tm;
	TickMeter tm_now;

	float pTime;
	float t_min = 0.0f, t_max = 0.0f;
	float t_ave = 0.0f;

	// memory of weighting function
	float *w = new float[nx * 8];
	memset(w, 0, nx * 8 * sizeof(float));

	// name of interpolation method
	//const char *wf = "Bilinear";
	//const char *wf = "Bi-cubic_4";
	//const char *wf = "Bi-cubic_6";
	//const char *wf = "Bi-cubic_8";
	//const char *wf = "Lagrange";
	const char *wf = "Bspline";

	wInter(nx, ny, wf, w);			// calculate weighting function

	for (int iter = 0; iter < iteration; iter++) {

		cout << "iteration number " << iter + 1 << " ";

		tm.push_back(tm_now);
		tm.at(iter).start();

		//Interp(src.data, height, width, wf, w, nx, ny, dst.data);		// serial interpolation
		Interp_omp(src.data, height, width, wf, w, nx, ny, dst.data);	// omp interpolation

		tm.at(iter).stop();
		pTime = tm.at(iter).getTimeMilli();
		printf("processing time : %.3f ms\n", pTime);

		t_ave += pTime;

		if (iter == 0)
		{
			t_min = pTime;
			t_max = pTime;
		}
		else
		{
			if (pTime < t_min) t_min = pTime;
			if (pTime > t_max) t_max = pTime;
		}
	}

	if (iteration == 1) t_ave = t_ave;
	else if (iteration == 2) t_ave = t_ave / 2;
	else t_ave = (t_ave - t_min - t_max) / (iteration - 2);

	cout << endl << "Average processing time : " << (float)t_ave << " ms" << endl;

	//imshow("src", src);
	//imshow("dst", dst);

	//waitKey(0);

	return 0;

}

// calculate weighting function
void wInter(int x, int y, const char * wf, float* w)
{
	// Bilinear
	if (strcmp(wf, "Bilinear") == 0) {
		x = x - 1;
		y = y - 1;
		int i;

		for (i = 0; i < x; i++)
		{
			w[i * 2 + 0] = 1 - (float)(i + 1) / (float)(x + 1);
			w[i * 2 + 1] = (float)(i + 1) / (float)(x + 1);
		}
	}
	// Bi-cubic_4
	else if (strcmp(wf, "Bi-cubic_4") == 0) {
		float a = -0.5f;
		float d;
		for (int i = 0; i < x; i++) {
			d = (float)(i + 1) / (float)x;
			// 0 <= |d| < 1
			w[i * 4 + 1] = (a + 2) * abs(pow(d, 3)) - (a + 3) * abs(pow(d, 2)) + 1;
			w[i * 4 + 2] = (a + 2) * abs(pow((1 - d), 3)) - (a + 3) * abs(pow((1 - d), 2)) + 1;
			// 1 <= |d| < 2
			w[i * 4 + 0] = a * abs(pow(d + 1, 3)) - 5 * a * abs(pow(d + 1, 2)) + 8 * a * abs(d + 1) - 4 * a;
			w[i * 4 + 3] = a * abs(pow((1 - d) + 1, 3)) - 5 * a * abs(pow((1 - d) + 1, 2)) + 8 * a * abs(((1 - d) + 1)) - 4 * a;
		}
	}
	// Bi-cubic_6
	else if (strcmp(wf, "Bi-cubic_6") == 0) {
		float d;
		for (int i = 0; i < x; i++) {
			d = (float)(i + 1) / (float)x;
			// 0 <= |d| < 1
			w[i * 6 + 2] = (6.f / 5.f) * abs(pow(d, 3)) + (-1.f) * (11.f / 5.f) * abs(pow(d, 2)) + 1;
			w[i * 6 + 3] = (6.f / 5.f) * abs(pow((1 - d), 3)) + (-1.f) * (11.f / 5.f) * abs(pow((1 - d), 2)) + 1;
			// 1 <= |d| < 2
			w[i * 6 + 1] = -1.f * (3.f / 5.f) * abs(pow(d + 1, 3)) + (16.f / 5.f)* abs(pow(d + 1, 2)) + (-1.f) * (27.f / 5.f) * abs((d + 1)) + (14.f / 5.f);
			w[i * 6 + 4] = -1.f * (3.f / 5.f) * abs(pow((1 - d) + 1, 3)) + (16.f / 5.f) * abs(pow((1 - d) + 1, 2)) + (-1.f) * (27.f / 5.f) * abs(((1 - d) + 1)) + (14.f / 5.f);
			// 2 <= |d| < 3			
			w[i * 6 + 0] = (1.f / 5.f) * abs(pow(d + 2, 3)) + (-1.f) * (8.f / 5.f)* abs(pow(d + 2, 2)) + (21.f / 5.f) * abs((d + 2)) + (-1.f) * (18.f / 5.f);
			w[i * 6 + 5] = (1.f / 5.f) * abs(pow((1 - d) + 2, 3)) + (-1.f) * (8.f / 5.f) * abs(pow((1 - d) + 2, 2)) + (21.f / 5.f) * abs(((1 - d) + 2)) + (-1.f) * (18.f / 5.f);
		}
	}
	// Bi-cubic_8
	else if (strcmp(wf, "Bi-cubic_8") == 0) {
		float d;
		for (int i = 0; i < x; i++) {
			d = (float)(i + 1) / (float)x;
			// 0 <= |d| < 1
			w[i * 8 + 3] = (67.f / 56.f) * abs(pow(d, 3)) - (123.f / 56.f) * abs(pow(d, 2)) + 1;
			w[i * 8 + 4] = (67.f / 56.f) * abs(pow((1 - d), 3)) - (123.f / 56.f) * abs(pow((1 - d), 2)) + 1;
			// 1 <= |d| < 2
			w[i * 8 + 2] = -1.f * (33.f / 56.f) * abs(pow(d + 1, 3)) + (177.f / 56.f) * abs(pow(d + 1, 2)) - (75.f / 14.f) * abs((d + 1)) + (39.f / 14.f);
			w[i * 8 + 5] = -1.f * (33.f / 56.f) * abs(pow((1 - d) + 1, 3)) + (177.f / 56.f) * abs(pow((1 - d) + 1, 2)) - (75.f / 14.f) * abs(((1 - d) + 1)) + (39.f / 14.f);
			// 2 <= |d| < 3			
			w[i * 8 + 1] = (9.f / 56.f) * abs(pow(d + 2, 3)) - (75.f / 56.f) * abs(pow(d + 2, 2)) + (51.f / 14.f) * abs((d + 2)) - (45.f / 14.f);
			w[i * 8 + 6] = (9.f / 56.f) * abs(pow((1 - d) + 2, 3)) - (75.f / 56.f) * abs(pow((1 - d) + 2, 2)) + (51.f / 14.f) * abs(((1 - d) + 2)) - (45.f / 14.f);
			// 3 <= |d| < 4
			w[i * 8 + 0] = -1.f * (3.f / 56.f) * abs(pow(d + 3, 3)) + (33.f / 56.f) * abs(pow(d + 3, 2)) - (15.f / 7.f) * abs((d + 3)) + (18.f / 7.f);
			w[i * 8 + 7] = -1.f * (3.f / 56.f) * abs(pow((1 - d) + 3, 3)) + (33.f / 56.f) * abs(pow((1 - d) + 3, 2)) - (15.f / 7.f) * abs(((1 - d) + 3)) + (18.f / 7.f);
		}
	}
	// Lagrange
	else if (strcmp(wf, "Lagrange") == 0) {
		float d;
		for (int i = 0; i < x; i++) {
			d = (float)(i + 1) / (float)x;
			// 0 <= |d| < 1
			w[i * 4 + 1] = (1.f / 2.f) * abs(pow(d, 3)) - abs(pow(d, 2)) - (1.f / 2.f) * abs(d) + 1;
			w[i * 4 + 2] = (1.f / 2.f) * abs(pow((1 - d), 3)) - abs(pow((1 - d), 2)) - (1.f / 2.f) * abs((1 - d)) + 1;
			// 1 <= |d| < 2
			w[i * 4 + 0] = -1.f * (1.f / 6.f) * abs(pow(d + 1, 3)) + abs(pow(d + 1, 2)) - (11.f / 6.f) * abs((d + 1)) + 1;
			w[i * 4 + 3] = -1.f * (1.f / 6.f) * abs(pow((1 - d) + 1, 3)) + abs(pow((1 - d) + 1, 2)) - (11.f / 6.f) * abs(((1 - d) + 1)) + 1;
		}
	}
	// Bspline
	else if (strcmp(wf, "Bspline") == 0) {
		float d; // 0 <= d < 1
		for (int i = 0; i < x; i++) {
			d = (float)(i + 1) / (float)x;	// 1/3 , 2/3
			// 0 <= |d| < 1
			w[i * 4 + 1] = (1.f / 2.f) * abs(pow(d, 3)) - abs(pow(d, 2)) + (2.f / 3.f);
			w[i * 4 + 2] = (1.f / 2.f) * abs(pow((1 - d), 3)) - abs(pow((1 - d), 2)) + (2.f / 3.f);
			// 1 <= |d| < 2
			w[i * 4 + 0] = -1.f * (1.f / 6.f) * abs(pow(d + 1, 3)) + abs(pow(d + 1, 2)) - 2.f * abs((d + 1)) + (4.f / 3.f);
			w[i * 4 + 3] = -1.f * (1.f / 6.f) * abs(pow((1 - d) + 1, 3)) + abs(pow((1 - d) + 1, 2)) - 2.f * abs(((1 - d) + 1)) + (4.f / 3.f);
		}
	}
}

void Interp(unsigned char * src, int hg, int wd, const char * wf, float *w, int x, int y, unsigned char * output) {

	x = x - 1;
	y = y - 1;
	int r, c, i, j, nc, nr, size;
	int nwd = wd * (x + 1);
	float temp;

	// check size according on method
	if (strcmp(wf, "Bilinear") == 0)		size = 1;
	else if (strcmp(wf, "Bi-cubic_6") == 0)	size = 3;
	else if (strcmp(wf, "Bi-cubic_8") == 0)	size = 4;
	else									size = 2;

	for (r = 0; r < hg; r++) {
		for (c = 0 + size - 1; c < wd - size; c++)
		{
			nr = r * (y + 1);
			nc = c * (x + 1);

			output[nr*nwd + nc] = src[r*wd + c];

			for (i = 0; i < x; i++)
			{
				nc = c * (x + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
					temp += w[i*(size * 2) + j] * (float)src[r*wd + c - size + j + 1];

				output[nr*nwd + nc] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}

	int ntemp;

	for (r = 0 + size - 1; r < hg - size; r++) {
		for (c = 0 * (x + 1); c < wd*(x + 1) + x; c++)
		{
			for (i = 0; i < y; i++)
			{
				nr = r * (y + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
				{
					ntemp = (r - size + j + 1)*(y + 1);
					temp += w[i*(size * 2) + j] * (float)output[ntemp*nwd + c];
				}

				output[nr*nwd + c] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}
}

void Interp_omp(unsigned char * src, int hg, int wd, const char * wf, float *w, int x, int y, unsigned char * output) {

	x = x - 1;
	y = y - 1;

	// int r, c, i, j, nc, nr;	// In openmp, the variables below should be different from section to section,
								// so do not declare in advance.	
	int size;
	int nwd = wd * (x + 1);
	float temp;
	int ntemp;

	if (strcmp(wf, "Bilinear") == 0)		size = 1;
	else if (strcmp(wf, "Bi-cubic_6") == 0)	size = 3;
	else if (strcmp(wf, "Bi-cubic_8") == 0)	size = 4;
	else									size = 2;

	// copies variables and makes them local variables of threads
	// Divide into four horizontal and vertical sections.
#pragma omp parallel sections private(temp, ntemp)
	{
		// --------------------------- HORIZON ---------------------------
		// In the horizontal direction, the vertical edge shall be removed

#pragma omp section
		// 1st horizontal section
		for (int r = 0; r < hg / 4; r++)		// ( 0 ) ~ ( height / 4 )
		{
			for (int c = 0 + size - 1; c < wd - size; c++)
			{
				int nr = r * (y + 1);
				int nc = c * (x + 1);

				output[nr*nwd + nc] = src[r*wd + c];

				for (int i = 0; i < x; i++)
				{
					nc = c * (x + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
						temp += w[i*(size * 2) + j] * (float)src[r*wd + c - size + j + 1];

					output[nr*nwd + nc] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
		// 2nd horizontal section
#pragma omp section
		for (int r = hg / 4; r < hg / 2; r++)		// ( height / 4 ) ~ ( height / 2 )
		{
			for (int c = 0 + size - 1; c < wd - size; c++)
			{
				int nr = r * (y + 1);
				int nc = c * (x + 1);

				output[nr*nwd + nc] = src[r*wd + c];

				for (int i = 0; i < x; i++)
				{
					nc = c * (x + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
						temp += w[i*(size * 2) + j] * (float)src[r*wd + c - size + j + 1];

					output[nr*nwd + nc] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
		// 3rd horizontal section
#pragma omp section
		for (int r = hg / 2; r < 3 * hg / 4; r++)	// ( height / 2 ) ~ ( 3 * height / 4 )
		{
			for (int c = 0 + size - 1; c < wd - size; c++)
			{
				int nr = r * (y + 1);
				int nc = c * (x + 1);

				output[nr*nwd + nc] = src[r*wd + c];

				for (int i = 0; i < x; i++)
				{
					nc = c * (x + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
						temp += w[i*(size * 2) + j] * (float)src[r*wd + c - size + j + 1];

					output[nr*nwd + nc] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
		// 4th horizontal section
#pragma omp section
		for (int r = 3 * hg / 4; r < hg; r++)	// ( 3 * height / 4 ) ~ ( height )
		{
			for (int c = 0 + size - 1; c < wd - size; c++)
			{
				int nr = r * (y + 1);
				int nc = c * (x + 1);

				output[nr*nwd + nc] = src[r*wd + c];

				for (int i = 0; i < x; i++)
				{
					nc = c * (x + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
						temp += w[i*(size * 2) + j] * (float)src[r*wd + c - size + j + 1];

					output[nr*nwd + nc] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
	}

#pragma omp parallel sections private(temp, ntemp)
	{
// --------------------------- Vertical ---------------------------
// In the vertical direction, the horizontal edge shall also be removed

		// 1st vertical section
#pragma omp section
		for (int r = 0 + size - 1; r < hg / 4; r++) {		// ( size - 1 ) ~ ( height / 4 )
			for (int c = 0 * (x + 1); c < wd*(x + 1) + x; c++)
			{
				for (int i = 0; i < y; i++)
				{
					int nr = r * (y + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
					{
						ntemp = (r - size + j + 1)*(y + 1);
						temp += w[i*(size * 2) + j] * (float)output[ntemp*nwd + c];
					}

					output[nr*nwd + c] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
		// 2nd vertical section
#pragma omp section
		for (int r = hg / 4; r < hg / 2; r++) {				// ( height / 4 ) ~ ( height / 2 )
			for (int c = 0 * (x + 1); c < wd*(x + 1) + x; c++)
			{
				for (int i = 0; i < y; i++)
				{
					int nr = r * (y + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
					{
						ntemp = (r - size + j + 1)*(y + 1);
						temp += w[i*(size * 2) + j] * (float)output[ntemp*nwd + c];
					}

					output[nr*nwd + c] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
		// 3rd vertical section
#pragma omp section
		for (int r = hg / 2; r < 3 * hg / 4; r++) {			// ( height / 2 ) ~ ( 3 * height / 4 )
			for (int c = 0 * (x + 1); c < wd*(x + 1) + x; c++)
			{
				for (int i = 0; i < y; i++)
				{
					int nr = r * (y + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
					{
						ntemp = (r - size + j + 1)*(y + 1);
						temp += w[i*(size * 2) + j] * (float)output[ntemp*nwd + c];
					}

					output[nr*nwd + c] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
		// 4th vertical section
#pragma omp section
		for (int r = 3 * hg / 4; r < hg - size; r++) {		// ( 3 * height / 4 ) ~ ( hg - size )
			for (int c = 0 * (x + 1); c < wd*(x + 1) + x; c++)
			{
				for (int i = 0; i < y; i++)
				{
					int nr = r * (y + 1) + i + 1;
					temp = 0;
					for (int j = 0; j < size * 2; j++)
					{
						ntemp = (r - size + j + 1)*(y + 1);
						temp += w[i*(size * 2) + j] * (float)output[ntemp*nwd + c];
					}

					output[nr*nwd + c] = (unsigned char)((int)(temp + 0.5));
				}
			}
		}
	}
}