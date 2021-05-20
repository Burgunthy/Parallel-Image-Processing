#include <omp.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define n1 1000000
#define n2 100000

int main() {
	TickMeter tm1, tm2, tm3, tm4, tm5;

	omp_set_num_threads(16);

	int i = 0;
	int isum = 0;

	int j = 1;
	double dsum = 0;

// --------------------- Serial
	tm1.start();

	for (i = 0; i < n1; i++) isum += i;
	for (j = 1; j < n2; j++) dsum += j;

	tm1.stop();
	printf("processing time : %f msec\n", tm1.getTimeMilli());

	cout << isum << endl;
	cout << dsum << endl;

// --------------------- OpenMP

	isum = 0;
	dsum = 0.0;

	tm2.start();

#pragma omp parallel
	{
#pragma omp for reduction(+:isum)
		for (i = 0; i < n1; i++) isum += i;
#pragma omp for reduction(+:dsum)
		for (j = 1; j < n2; j++) dsum += j;
	}

	tm2.stop();
	printf("\nprocessing time : %f msec\n", tm2.getTimeMilli());
	
	cout << isum << endl;
	cout << dsum << endl;


// --------------------- OpenMP

	isum = 0;
	dsum = 0.0;

	tm4.start();

#pragma omp parallel for reduction(+:isum)
	for (i = 0; i < n1; i++) isum += i;

#pragma omp parallel for reduction(+:dsum)
	for (j = 1; j < n2; j++) dsum += j;

	tm4.stop();
	printf("\nprocessing time : %f msec\n", tm4.getTimeMilli());

	cout << isum << endl;
	cout << dsum << endl;

// --------------------- OpenMP

	isum = 0;
	dsum = 0.0;

	tm3.start();

#pragma omp parallel sections
	{
#pragma omp section
		for (i = 0; i < n1; i++) isum += i;
#pragma omp section
		for (j = 1; j < n2; j++) dsum += j;			// for¹® ÁßÃ¸
	}

	tm3.stop();
	printf("\nprocessing time : %f msec\n", tm3.getTimeMilli());

	cout << isum << endl;
	cout << dsum << endl;

	return 0;
}