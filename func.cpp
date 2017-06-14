 #include <iostream>
#include <vector>

#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "func.h";

using namespace std;
using namespace cv;


//расчет ускорения
float acceleration(Vec3f vec) {
	double summa = 0;

	for (int i = 0; i < vec.rows; i++) {
		summa = summa + pow(vec[i], 2);
	}

	return sqrt(summa);
}

//максимальное значение
float max_a(Mat vec) {
	float mmax = vec.at<float>(0, 0);

	for (int i = 1; i < vec.cols; i++)
		if (vec.at<float>(0, i) > mmax)
			mmax = vec.at<float>(0, i);

	return mmax;
}

//минимальное значение
float min_a(Mat vec) {
	float mmin = vec.at<float>(0, 0);

	for (int i = 1; i < vec.cols; i++)
		if (vec.at<float>(0, i) < mmin)
			mmin = vec.at<float>(0, i);

	return mmin;
}

//максимальный отрезок
float max_range(Mat vec) {
	float mmax = 0;
	float mrazn = 0;

	for (int i = 0; i < vec.cols - 1; i++) {
		mrazn = abs(vec.at<float>(0, i) - vec.at<float>(0, i + 1));

		if (mrazn > mmax)
			mmax = mrazn;
	}

	return mmax;
}

//мат. ожидание
float math_exp(Mat vec) {
	float expect = 0;

	for (int i = 0; i < vec.cols; i++)
		expect = expect + vec.at<float>(0, i);

	return expect / (vec.cols + 1);
}

//дисперсия
float dispersion(Mat vec) {
	Mat vec2(vec.rows, vec.cols, CV_32FC1);
	
	for (int i = 0; i < vec.cols; i++)
		vec2.at<float>(0, i) = pow(vec.at<float>(0, i), 2);

	return sqrt(math_exp(vec2) - math_exp(vec));
}

//определение участков ускорения/торможения
void get_land(Mat acc, vector<vector<float>> land, vector<int> & style) {
	vector<float> tmp;
	//vector<vector<float>> ltmp;
	//vector<int> st;

	int st = 0;

	tmp.push_back(acc.at<float>(1, 0));

	for (int i = 1; i < acc.cols; i++) {
		if (acc.at<float>(1, i) > tmp[i - 1]) { // >, max - 1
			if (i == 1) {
				tmp.push_back(acc.at<float>(1, i));
			}
			else if (i == 2) {
				land.push_back(tmp);
				tmp.clear();
				tmp.push_back(acc.at<float>(1, i));
				style.push_back(st);
			}

			st = 1;
		}
		else if (acc.at<float>(1, i) < tmp[i - 1]) { // <, min - 2
			if (i == 1) {
				land.push_back(tmp);
				tmp.clear();
				tmp.push_back(acc.at<float>(1, i));
				style.push_back(st);
			}
			else if (i == 2) {
				tmp.push_back(acc.at<float>(1, i));
			}

			st = 2;
		}

		if (!tmp.empty()) {
			land.push_back(tmp);
			style.push_back(st);
		}

	}
}

//преобразование Фурье
Mat DFT(Mat I) {
	//Mat I;

	Mat padded; 
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI, DFT_COMPLEX_OUTPUT);

	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	//magI += Scalar::all(1);                    // switch to logarithmic scale
	//log(magI, magI);

	//magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	//normalize(magI, magI, 0, 1, CV_MINMAX);
	//normalize(magI, magI, 0, 1, CV_MINMAX);

	return magI;
}

Mat vecF(Mat I) {
	Mat vec(1, I.rows, CV_32FC1);
	//vector<float> vec;
	float summa;
	
	for (int i = 0; i < I.rows; i++) {
		summa = 0;
		for (int j = 0; j < I.cols; j++)
			summa += I.at<float>(i, j);
		
		vec.at<float>(0, i) = summa;
	}

	return vec;
}

Mat kef(Mat I, int n) {
	Mat vec(1, n, CV_32FC1);

	int i = 0;
	int j = 1;
	while (i < n) {
		vec.at<float>(0, i) = I.at<float>(0, j);
		vec.at<float>(0, i + 1) = I.at<float>(0, I.cols - j);
		i += 2;
		j += 1;
	}
	return vec;
}

Mat kef2(Mat I, int n) {
	Mat vec = I(Range(0, I.rows), Range(1, n + 1)); 

	return vec;
}