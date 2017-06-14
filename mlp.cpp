#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <vector>
#include <opencv2\core\core.hpp>
#include <opencv2\ml\ml.hpp>

#include "mlp.h"

using namespace std;
using namespace cv;



Mat labelData(Mat input, vector<int> id, int n) {
	Mat output(input.rows, n, CV_32FC1);
	
	for (int i = 0; i < input.rows; i++) {
		//Mat row = input.row(i).clone();
		int nid = id[i];
		for (int j = 0; j < output.cols; ++j)
			output.at <float>(i, j) = -0.4;
		//if (nid != -1) 
			output.at <float>(i, nid - 1) = 0.8;
	}
	return output;
}

//определяет водителя
int type_driver(Mat pred) {
	int number = 0;
	float max = pred.at<float>(0, 0);
	for (int i = 1; i < pred.cols; ++i) {
		if (pred.at<float>(0, i) > max) {
			max = pred.at<float>(0, i);
			number = i;
		}
	}
	if (max < 0.1) return -1;
	else return number + 1 ;
}

void print_class(int n) {
	switch (n) {
	case 1: cout << "1. Driver " << endl; break;
	case 2: cout << "2. Driver " << endl; break;
	case 3: cout << "3. Driver " << endl; break;
	case 4: cout << "4. Driver " << endl; break;
	case 5: cout << "5. Driver " << endl; break;
	case 6: cout << "6. Driver " << endl; break;
	case 7: cout << "7. Driver " << endl; break;
	case 8: cout << "8. Driver " << endl; break;
	case 9: cout << "9. Driver " << endl; break;
	case 10: cout << "10. Driver " << endl; break;
	default: cout << " Водитель не определен! "<< endl;
	}
}

float evaluate(Mat & predicted, Mat& actual) {

	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at <float >(i, 0);
		float a = actual.at <float >(i, 0);
		if ((p >= 0.1 && a >= 0.1) || (p <= 0.1 && a <= 0.1)) {
			t++;
		}
		else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}

void perceptron(Mat trainingData, Mat testData, vector<int> trainingID, vector<int> testID, int in, int out)
{
	//кол-во входов
	//int in = 4;

	//кол-во выходов
	//int out = 5;

	Mat trainingClasses = labelData(trainingData, trainingID, out);
	
	Mat testClasses = labelData(testData, testID, out);
	
	//  Определяем структуру сети
	Mat layers = Mat(4, 1, CV_32SC1);

	//  Входной слой - 5 сенсоров
	layers.row(0) = Scalar(in);
	layers.row(1) = Scalar(45);
	layers.row(2) = Scalar(45);
	layers.row(3) = Scalar(out);

	CvANN_MLP mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 700;
	criteria.epsilon = 0.001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1f;
	params.bp_moment_scale = 0.1f;
	params.rp_dw0 = 0.1;
	//params.rp_dw_min = FLT_EPSILON;
	//params.term_crit = criteria;
	cout << "Params inited\n";
	mlp.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);
	cout << "Perseptron created...\n";

	cout << "Training...\n";
	mlp.train(trainingData, trainingClasses, Mat(), Mat(), params);
	cout << "Trained!\n";

	Mat response(1, out, CV_32FC1);
	Mat predicted(testClasses.rows, out, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		Mat response(1, out, CV_32FC1);
		Mat sample = testData.row(i);
		mlp.predict(sample, response);
		for (int j = 0; j < response.cols; ++j)
			predicted.at <float>(i, j) = response.at <float>(0, j);
	}

	cout << "Testing complete.\n";
	cout << " Accuracy_ {MLP} = " << evaluate(predicted, testClasses) << endl;
	for (int i = 0; i < predicted.rows; i++) {
		for (int j = 0; j < predicted.cols; ++j)
			cout << predicted.at <float>(i, j) << " ";
	}
	cout << endl;
	int typ = type_driver(predicted);
	print_class(typ);
}