#pragma once
#include "HeadPose.h"

class BrownDistortion
{
protected:
	//Camera Intrinsic
	double fx;
	double fy;
	double cx;
	double cy;
	//Distortion Coefficient
	double k1;
	double k2;
	double k3;
	double p1;
	double p2;
	//Camera Parameter and points2d
	Vector<double, 5> distcoeff;
	Matrix3d intrinsic;
	vector<Vector2d> poins2d;
	
	
public:
	BrownDistortion() {};

	pair<Matrix3d, Vector5d> GetInputData(vector<Vector2d> _poins2d, Matrix3d _intrinsic, Vector5d _distcoeff);

	//void GetValue();

	vector<Vector2d> Distortion();

	vector<Vector2d> Undistortion(pair<Matrix3d, Vector5d> IntrDist);

	~BrownDistortion() {};
};

//class BrownUndistortion:public BrownDistortion
//{
//private:
//	double eps = 1e-12;
//public:
//	BrownUndistortion() {};
//
//	
//
//	~BrownUndistortion() {};
//};