#pragma once

#include <io.h>
#include <vector>
#include <math.h>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <direct.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#define M_PI 3.1415926535898

using namespace std;
using namespace Eigen;
using namespace Sophus;

typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 1, 5> Vector5d;


