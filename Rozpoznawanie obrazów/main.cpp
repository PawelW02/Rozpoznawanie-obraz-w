#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <sys/timeb.h>
#include <vector>
#include <string>
#include <windows.h>
#include <fstream>
#include "nlohmann\json.hpp"

using json = nlohmann::json;
using namespace cv;
using namespace std;
//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
//{
//	MessageBox(NULL, "RADIOWÓZ ZA TOBĄ!", "Uwaga", MB_ICONWARNING | MB_OK);
//	return 0;
//}

int main() {

	////////// VARIABLES //////////

	cv::VideoCapture cap(1);
	cv::Mat img;
	cv::CascadeClassifier plateCascade;
	std::vector<cv::Rect> plate;				
	std::string outText;
	tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

	////////// PLATES DATABASE //////////

	// Otwieranie pliku JSON
	std::ifstream jsonFile("example.json");
	json jsonData;
	jsonFile >> jsonData;


	////////// IMPORT FILES TO RECOGNITION //////////
	
	plateCascade.load("haarcascade_plate_number.xml");

	if (plateCascade.empty()) { std::cout << "XML file not loaded" << std::endl; }

	////////// OCR //////////

	api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
	api->SetPageSegMode(tesseract::PSM_AUTO);


	////////// CAM OPENING //////////

	while (true) {

		cap.read(img);

		Mat imgLaplacian, imgResult;
		plateCascade.detectMultiScale(img, plate, 1.1, 10);
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		Mat plateImg = img(plate[0]);
		// ok, now try different kernel
		Mat kernel = (Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1); // another approximation of second derivate, more stronger

		// do the laplacian filtering as it is
		// well, we need to convert everything in something more deeper then CV_8U
		// because the kernel has some negative values, 
		// and we can expect in general to have a Laplacian image with negative values
		// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
		// so the possible negative number will be truncated
		filter2D(plateImg, imgLaplacian, CV_32F, kernel);
		plateImg.convertTo(plateImg, CV_32F);
		imgResult = plateImg - imgLaplacian;

		// convert back to 8bits gray scale
		imgResult.convertTo(imgResult, CV_8U);
		imgLaplacian.convertTo(imgLaplacian, CV_8U);

		plateImg = imgResult;



		PIX* pixS = pixCreate(plateImg.size().width, plateImg.size().height, 8);


		for (int i = 0; i < plateImg.rows; i++)
			for (int j = 0; j < plateImg.cols; j++)
				pixSetPixel(pixS, j, i, (l_uint32)plateImg.at<uchar>(i, j));


		api->SetImage(pixS);

		outText = api->GetUTF8Text();
		std::cout << outText << "\n";

		//for (int i = 0; i < plate.size(); i++)
		//	{
		//		//cv::Mat imgCrop = img(plate[i]);
		//		//imshow(std::to_string(i), imgCrop);
		//		


		//		//cv::imwrite("Resources/Plates/" + std::to_string(i) + ".png", imgCrop);
		//		std::stringstream s;
		//		s << outText << std::flush;
		//		cv::rectangle(img, plate[i].tl(), plate[i].br(), cv::Scalar(255, 0, 255), 3);
		//		cv::putText(img, s.str().c_str(), plate[i].tl(), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 0));

		//		//bool found = false;
		//		//for (auto& w : jsonData) {
		//		//	if (w == outText) {
		//		//		found = true;
		//		//		//break;
		//		//	}
		//		//}

		//		if (jsonData.contains(outText)) {
		//			std::cout << "Tablica rejestracyjna: " << outText << " znajduje się w bazie." << std::endl;
		//			Sleep(3000);
		//		}
		//		else {
		//			std::cout << "Słowo nie znajduje się w pliku JSON. Rozpoznane słowo: "<< outText << std::endl;
		//		}

		//		//std::cout << outText << ", " << jsonData;
		//	}

		cv::imshow("Kamera", img);
		cv::waitKey(1);

	}
	api->End();

	return 0;
}