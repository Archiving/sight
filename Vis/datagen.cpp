
/* datagen.cpp creates 2 sets of data: images and classifications */
/* KNN Character Recognition */

#include "stdafx.h"
#include "datagen.h"

using namespace cv;

//temporary
int data_gen() {
	Mat trainingNumbers;
	Mat imgGreyscale, imgBlurred, imgThresh, imgThreshCopy;
	Mat classificationInts; //training classifications
	Mat trainingImagesAsFlattenedFloats;

	trainingNumbers = imread("image.png", CV_LOAD_IMAGE_COLOR);
	if (!trainingNumbers.data) {
		std::cout << "Could not read file!" << std::endl;
		std::cin.get();
		return 0;
	}
	std::vector<int> valid = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	std::vector<Vec4i> v4iH; //contours hierarchy
	std::vector<std::vector<Point>> ptContours; //contours vector

	//Greyscale image
	cvtColor(trainingNumbers, imgGreyscale, CV_BGR2GRAY);
	//Blur
	GaussianBlur(imgGreyscale, imgBlurred, Size(5, 5), 0);
	//Threshold
	adaptiveThreshold(imgBlurred, imgThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

	imgThreshCopy = imgThresh.clone();
	findContours(imgThreshCopy, ptContours, v4iH, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < ptContours.size(); i++) {
		if (contourArea(ptContours[i]) >= MIN_CONTOUR_AREA) {
			Rect boundingRect = cv::boundingRect(ptContours[i]);
			cv::rectangle(trainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);

			Mat ROI = imgThresh(boundingRect);
			Mat resizedROI;
			cv::resize(ROI, resizedROI, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), 2);

			imshow("ROI", ROI);
			imshow("ResizedROI", resizedROI);
			imshow("Training", trainingNumbers);

			int intChar = waitKey(0);
			if (intChar == 27) {
				return 0;
			}
			else if (std::find(valid.begin(), valid.end(), intChar) != valid.end()) {
				classificationInts.push_back(intChar);
				Mat imageFloat; //now add the training image
				resizedROI.convertTo(imageFloat, CV_32FC1);
				Mat imageFlattenedFloat = imageFloat.reshape(1, 1);
				trainingImagesAsFlattenedFloats.push_back(imageFlattenedFloat);
			}
			else {
				std::cout << "Not a valid character." << std::endl;
			}
		}
	}

	std::cout << "Training complete!\n";

	//conversion and write to file
	FileStorage classifications("classifications.xml", FileStorage::WRITE);
	if (!classifications.isOpened()) {
		std::cout << "Couldnt write classifications.xml to file, exiting program" << std::endl;
		std::cin.get();
		return 0;
	}

	classifications << "classifications" << classificationInts;
	classifications.release();

	FileStorage trainingImages("images.xml", FileStorage::WRITE);
	if (!trainingImages.isOpened()) {
		std::cout << "Couldn't write images.xml to file, exiting program" << std::endl;
		std::cin.get();
		return 0;
	}
	trainingImages << "images" << trainingImagesAsFlattenedFloats;
	trainingImages.release();

	waitKey(0);
	return 0;
}