#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <vector>
#include <opencv2\core\core.hpp>
#include <opencv2\ml\ml.hpp>

using namespace std;
using namespace cv;

Mat labelData(Mat input, vector<int> ID, int n);
int type_driver(Mat pred);
void print_class(int n);
float evaluate(Mat & predicted, Mat& actual);
void perceptron(Mat trainingData, Mat testData, vector<int> trainingID, vector<int> testID, int inlay, int out);

#endif