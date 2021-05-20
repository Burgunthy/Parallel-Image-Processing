#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

Mat frame;								// live frame
string strVideo = "vvv.mp4";			// video name

CascadeClassifier face_cascade;
CascadeClassifier face_nested_cascade;
CascadeClassifier body_cascade;

string face_cascade_name =
	"D://study/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
string face_nested_cascade_name =
	"D://study/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string body_cascade_name =
	"D://study/opencv/sources/data/haarcascades/haarcascade_fullbody.xml";



int original_frame(VideoCapture cap) {

	while (1) {
		cap >> frame;		// Save the video in Mat format
		if (frame.empty()) {
			break;
		}

		waitKey(27);
	}

	return 0;
}

int DisplayVideo(string windowName) {
	
	Mat dst;								// output image
	namedWindow(windowName, 0);
	double fstart, fend, fprocTime;
	double fps;

	while (1) {
		fstart = omp_get_wtime();

		if (frame.empty()) {
			destroyWindow(windowName);
			break;
		}
		dst = frame.clone();				// copy frame to dst image

		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		// check fps and show text
		putText(dst, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);

		imshow(windowName, dst);		// Display Video

		if (waitKey(27) == 27)
			imwrite("original.png", dst);
	}

	return 0;
}

int GrabVideo(string windowName) {
	// Similar with DisplayVideo
	Mat dst;								// output image
	namedWindow(windowName, 0);
	double fstart, fend, fprocTime;
	double fps;

	while (1) {
		fstart = omp_get_wtime();

		if (frame.empty()) {
			destroyWindow(windowName);
			break;
		}
		dst = frame.clone();				// copy frame to dst image

		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		// check fps and show text
		putText(dst, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);

		imshow(windowName, dst);

		waitKey(0);					// If key input occur, grab the video and show it
	}

	return 0;
}

int detectFace(string windowName) {
	vector<Rect> faces;						// record faces

	Mat dst;								// output image
	Mat frame_gray;							// gray image for 
	namedWindow(windowName, 0);
	double fstart, fend, fprocTime;
	double fps;

	while (1) {
		fstart = omp_get_wtime();

		if (frame.empty()) {
			destroyWindow(windowName);
			break;
		}
		dst = frame.clone();				// copy frame to dst image
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);	// get gray image
		equalizeHist(frame_gray, frame_gray);			// normalize gray image

		// detect face
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t i = 0; i < faces.size(); i++)		// draw rectangle on all of faces
		{
			// draw red rectangle in face
			rectangle(dst, Point(faces[i].x, faces[i].y),
				Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
				Scalar(0, 0, 255), 3);

			Mat faceROI = frame_gray(faces[i]);			// minimize search space
			std::vector<Rect> eyes;						// record eyes

			// detect eyes in face range
			face_nested_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

			for (size_t j = 0; j < eyes.size(); j++) {	// draw rectangle on all of eyes

				// draw green rectangle in eye
				rectangle(dst, Point(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y),
					Point(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height),
					Scalar(0, 255, 0), 2);
			}
		}

		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		putText(dst, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);

		imshow(windowName, dst);
		if (waitKey(27) == 27)
			imwrite("face.png", dst);
	}

	return 0;
}

int detectBody(string windowName) {
	vector<Rect> bodies;					// record bodies

	Mat dst;								// output image
	Mat frame_gray;							// gray image for 
	namedWindow(windowName, 0);
	double fstart, fend, fprocTime;
	double fps;

	while (1) {
		fstart = omp_get_wtime();

		if (frame.empty()) {
			destroyWindow(windowName);
			break;
		}
		dst = frame.clone();				// copy frame to dst image
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);	// get gray image
		equalizeHist(frame_gray, frame_gray);			// normalize gray image

		// detect bodies
		body_cascade.detectMultiScale(frame_gray, bodies, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(70, 70));

		for (size_t i = 0; i < bodies.size(); i++) {	// draw rectangle on all of bodies

			// draw rectangle on all of bodies
			rectangle(dst, Point(bodies[i].x, bodies[i].y),
				Point(bodies[i].x + bodies[i].width, bodies[i].y + bodies[i].height),
				Scalar(255, 0, 255), 3);
		}

		fend = omp_get_wtime();
		fprocTime = fend - fstart;
		fps = (1 / (fprocTime));
		putText(dst, "fps : " + to_string(fps), Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 3);

		imshow(windowName, dst);
		if(waitKey(27) == 27)
			imwrite("body.png", dst);
	}

	return 0;
}

int main(int argc, char **argv) {
	// if there are not cascade code, return -1
	if (!face_cascade.load(face_cascade_name)) {				// check face detection
		printf("--(!)Error loading face cascade\n");
		return -1;
	}
	if (!face_nested_cascade.load(face_nested_cascade_name)) {	// check eye detection
		printf("--(!)Error loading eye cascade\n");
		return -1;
	}
	if (!body_cascade.load(body_cascade_name)) {				// check body detection
		printf("--(!)Error loading body cascade\n");
		return -1;
	}

	VideoCapture cap(0);					// capture online video
	if (!cap.isOpened()) return -1;
	cap >> frame;							// read first frame

#pragma omp parallel sections				// parallel processing
	{
#pragma omp section
		original_frame(cap);		// Read video and save Frame
#pragma omp section
		DisplayVideo("Display");			// Display Frame
#pragma omp section
		GrabVideo("Grab");					// Capture a Frame
#pragma omp section
		detectFace("Face");					// Detect Face and Eye
#pragma omp section
		detectBody("Body");					// Detect Body
	}

	return 0;
}