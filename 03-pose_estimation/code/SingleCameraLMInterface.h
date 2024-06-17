#pragma once
#include "HeadPose.h"

class LevenbergMarquardtSingle
{
private:
	SE3d pose;
	double epsilon1, epsilon2;
	double tau;
	int max_iter;
	bool is_out;
	vector<Vector3d> points3d;
	vector<Vector2d> points2d;
	Vector2d focal;
	Vector2d center;
	Eigen::MatrixXd e;
	Eigen::MatrixXd J;
	Eigen::Matrix<double, 6, 6> H;
	Vector6d g;
	Matrix4d p_ini;

	double residual_init_0;

public:
	LevenbergMarquardtSingle();

	void AddObservation(const vector<Vector3d>& points3d_, const vector<Vector2d>& points2d_,
		                Eigen::Matrix3d Intrinsic, Matrix4d p0, double residual_init);

	void Jacobian();
	
	void Hessian();

	double GetCost();

	double F(SE3d p);

	double L0L(Vector6d dx);

	void Solution(Eigen::Matrix<double, 7, 1>& qt, double& pixel_error, int& diverged);
};
