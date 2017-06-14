#ifndef FUNC_H
#define FUNC_H

#include <iostream>
#include <vector>
#include <opencv2\core\core.hpp>
#include <opencv2\ml\ml.hpp>

using namespace std;
using namespace cv;

//Mat inlayer(Mat input, int n);
float acceleration(Vec3f vec);
float max_a(Mat vec);
float min_a(Mat vec);
float max_range(Mat vec);
float math_exp(Mat vec);
float dispersion(Mat vec);
void get_land(Mat acc, vector<vector<float>> land, vector<int> & style);
Mat DFT(Mat I);
Mat vecF(Mat I);
Mat kef(Mat I, int n);
Mat kef2(Mat I, int n);

#endif