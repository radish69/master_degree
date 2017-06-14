#define _CRT_SECURE_NO_DEPRECATE

#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <string>
#include <ctime>

#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

#include "mlp.h"
#include "func.h"

using namespace std;
using namespace cv;

typedef float elemType;

static void read_csv(const string& filename, vector<double>& labels, Mat_<Vec3f>& cor, vector<int>& Devid, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, ms, x, y, z, id;
	Vec3f vec;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, ms, separator);
		getline(liness, x, separator);
		getline(liness, y, separator);
		getline(liness, z, separator);
		getline(liness, id, separator);

		vec[0] = atof(x.c_str());
		vec[1] = atof(y.c_str());
		vec[2] = atof(z.c_str());
		

		if (!ms.empty() && !x.empty() && !y.empty() && !z.empty() && !id.empty()) {
			labels.push_back(atof(ms.c_str()));
			cor.push_back(vec);
			Devid.push_back(atoi(id.c_str()));
		}
	}
}

int outputAcc(vector<float> gen, string name) {
	// св€зываем объект с файлом, при этом файл открываем в режиме записи, предварительно удал€€ все данные из него
	ofstream fout(name, ios_base::out | ios_base::trunc);

	if (!fout.is_open()) // если файл небыл открыт
	{
		cout << "‘айл не может быть открыт или создан\n";
		return 0;
	}

	for (int i = 0; i < gen.size(); i++) {
		fout << gen[i] << "|" << endl;;
	}

	fout.close();
	return 1;
}

int outputM(Mat I, string name) {
	// св€зываем объект с файлом, при этом файл открываем в режиме записи, предварительно удал€€ все данные из него
	ofstream fout(name, ios_base::out | ios_base::trunc);
	int a = 0;
	int b = I.rows;

	if (!fout.is_open()) // если файл небыл открыт
	{
		cout << "‘айл не может быть открыт или создан\n";
		return 0;
	}

	for (int i = a; i < b; i++) {
		for (int j = 0; j < I.cols; j++)
			fout << I.at<float>(i, j) << "|";
		fout << endl;
	}

	fout.close();
	return 1;
}

int outputTest(Mat I, vector<int> v, string name) {
	// св€зываем объект с файлом, при этом файл открываем в режиме записи, предварительно удал€€ все данные из него
	ofstream fout(name, ios_base::out | ios_base::trunc);
	int a = 0;
	int b = I.rows;

	if (!fout.is_open()) // если файл небыл открыт
	{
		cout << "‘айл не может быть открыт или создан\n";
		return 0;
	}

	for (int i = a; i < b; i++) {
		for (int j = 0; j < I.cols; j++)
			fout << I.at<float>(i, j) << "|";
		fout << " - " << v[i];
		fout << endl;
	}

	fout.close();
	return 1;
}

vector<float> vec_acc(Mat_<Vec3f> I){
	vector<float> gen;

	for (int i = 0; i < I.rows; i++)
		gen.push_back(acceleration(I.at<Vec3f>(i)));

	return gen;
}

vector<float> acctojerk(vector<float> acc) {
	vector<float> gen;
	int i = 1;
	gen.push_back(0);

	while (i < acc.size()) {
		if (!(i%500)) {
			gen.push_back(0);
		}
		else {
			gen.push_back(acc[i] - acc[i - 1]);
		}
		i++;
	}

	return gen;
}

Mat statch(Mat I) {
	Mat gen(I.rows, 5, CV_32FC1);

	for (int i = 0; i < gen.rows; i++) {
		gen.at<float>(i, 0) = max_a(I.row(i));
		gen.at<float>(i, 1) = min_a(I.row(i));
		gen.at<float>(i, 2) = max_range(I.row(i));
		gen.at<float>(i, 3) = math_exp(I.row(i));
		gen.at<float>(i, 4) = dispersion(I.row(i));
	}

	return gen;
}

void firstgroup(Mat I, Mat T, vector<int> id, vector<int> idT, int inlay, int outlay) {
	Mat trainingData = statch(I);

	Mat testData = statch(T);

	perceptron(trainingData, testData, id, idT, inlay, outlay);

	system("pause");
}


int main()
{
	setlocale(LC_ALL, "rus");

	vector<double> labels;
	Mat_<Vec3f> Acc;
	vector<int> id;
	
	//файл с данными
	string filename = "new.csv";

	//чтение данных
	read_csv(filename, labels, Acc, id);

	//переходим к ускорению
	vector<float> vAcc = vec_acc(Acc);

	//вывод в файл данных общего ускорени€
	int res = outputAcc(vAcc, "data_acc.txt");

	//примен€ем jerk-фильтр
	vector<float> jerk = acctojerk(vAcc);

	//вывод в файл фильтрованных данных
	res = outputAcc(jerk, "data_jerk.txt");

	//кол-во обрабатываемых строк
	int block = 50;
	//шаг
	int step = 25;

	//кол-во водителей
	int vodil = 2;

	//входной слой
	int inlay = 5;

	Mat vecAcc((jerk.size() / step - vodil), block, CV_32FC1);
	vector<int> idjerk;
	int ot, doo;
	for (int k = 0; k < vodil; k++){
		ot = k*vecAcc.rows / vodil;
		doo = (k + 1)*vecAcc.rows / vodil;

		for (int i = ot; i < doo; i++){
			for (int j = 0; j < vecAcc.cols; j++){
				vecAcc.at<float>(i, j) = jerk[step*i + j];
			}
			idjerk.push_back(k+1);
		}
	}

	res = outputM(vecAcc, "data_window.txt");

	int st = 10;
	int ro = 0;
	Mat vecFF(vecAcc.rows / st + 1, st, CV_32FC1);
	vector<int> idFF;
	for (int k = 0; k < vodil; k++) {
		ot = k*vecAcc.rows / vodil;
		doo = (k + 1)*vecAcc.rows / vodil;
		
		for (int i = ot; i < doo; i += st){
			Mat copyvecAcc;
			if (i + st > doo) {
				copyvecAcc = vecAcc(Range(i, doo), Range(0, vecAcc.cols));
			}
			else {
				copyvecAcc = vecAcc(Range(i, i + st), Range(0, vecAcc.cols));
			}
			Mat DFTAcc = DFT(copyvecAcc);
			Mat F = vecF(DFTAcc);
			for (int j = 0; j < F.cols; j++) {
				vecFF.at<float>(ro, j) = F.at<float>(0, j);
			}
			ro += 1;
			idFF.push_back(k+1);
			res = outputM(DFTAcc, "data_dft.txt");
		}
	}
	res = outputTest(vecFF, idFF, "data_tteesstt.txt");

	Mat coeff = kef2(vecFF, inlay);
	res = outputM(coeff, "data_coeff.txt");
		
	Mat trainingData = coeff;
	

	//организуем тесты
	vector<double> labelsT;
	Mat_<Vec3f> AccT;
	vector<int> idT;
	//файл с данными
	filename = "driver//dr2_10.csv";
	int answer = 2;
	//чтение данных
	read_csv(filename, labelsT, AccT, idT);

	vector<float> vAccT = vec_acc(AccT);
	vector<float> jerkT = acctojerk(vAccT);
	Mat vecAccT(jerkT.size() / step - 1, block, CV_32FC1);
	vector<int> idjerkT;

	for (int i = 0; i < vecAccT.rows; i++) {
		for (int j = 0; j < vecAccT.cols; j++) {
			vecAccT.at<float>(i, j) = jerkT[step*i + j];
		}
		idjerkT.push_back(answer);
	}

	//firstgroup(vecAcc, vecAccT, id, idT, inlay, outlay);
	//return 0;

	Mat vecFFT(vecAccT.rows / st, st, CV_32FC1);
	vector<int> idFFT;
	ro = 0;
	for (int i = 0; i < vecAccT.rows; i += st){
		Mat copyvecAcc = vecAccT(Range(0, vecAccT.rows), Range(0, vecAccT.cols));
		Mat DFTAccT = DFT(copyvecAcc);
		Mat FT = vecF(DFTAccT);
		for (int j = 0; j < FT.cols; j++) {
			vecFFT.at<float>(ro, j) = FT.at<float>(0, j);
		}
		ro += 1;
		idFFT.push_back(idjerkT[i]);
	}

	Mat coeffT = kef2(vecFFT, inlay);
	res = outputM(coeffT, "data_coeffT.txt");
	Mat testData = coeffT;

	perceptron(trainingData, testData, idFF, idT, inlay, vodil);
	
	system("pause");
	
	return 0;
}