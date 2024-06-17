#pragma once
#include "SingleCameraLMInterface.h"

class Runtimer
{
private:
	std::chrono::steady_clock::time_point ts;
	std::chrono::steady_clock::time_point te;
public:
	Runtimer() {};
	~Runtimer() {};

	inline void start()
	{
		ts = std::chrono::steady_clock::now();
	}
	inline void stop()
	{
		te = std::chrono::steady_clock::now();
	}
	inline double duration()
	{
		return std::chrono::duration_cast<std::chrono::duration<double>>(te - ts).count() * 1000.0;
	}
};

LevenbergMarquardtSingle::LevenbergMarquardtSingle()
{
	epsilon1 = 1e-2;
	epsilon2 = 1e-12;
	tau = 1e-5;
	max_iter = 1000;
	is_out = true;
}

void LevenbergMarquardtSingle::AddObservation(const vector<Vector3d>& points3d_, const vector<Vector2d>& points2d_, 
	                                          Eigen::Matrix3d Intrinsic, Matrix4d p0, double residual_init)
{
	points3d.clear();
	points2d.clear();

	points3d.assign(points3d_.begin(), points3d_.end());
	points2d.assign(points2d_.begin(), points2d_.end());

	focal = Vector2d(Intrinsic(0, 0), Intrinsic(1, 1));
	center = Vector2d(Intrinsic(0, 2), Intrinsic(1, 2));

	Eigen::Matrix3d R; 
	p_ini = p0;
	R = p0.block(0, 0, 3, 3);
	Vector3d t = p0.block(0, 3, 3, 1);
	pose = Sophus::SE3d(R, t);
	residual_init_0 = residual_init;
}

void LevenbergMarquardtSingle::Jacobian()
{
	J.resize(2 * points2d.size(), 6);
	e.resize(2 * points2d.size(), 1);
	double fx = focal(0);
	double fy = focal(1);
	double cx = center(0);
	double cy = center(1);
	for (size_t i = 0; i < points2d.size(); i++)
	{
		Eigen::Vector3d pc = pose * points3d[i];
		double inv_z = 1.0 / pc[2];
		double inv_z2 = inv_z * inv_z;
		Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
		Eigen::Vector2d ei = points2d[i] - proj;
		Eigen::Matrix<double, 2, 6> Ji;
		Ji << -fx * inv_z, 0, fx* pc[0] * inv_z2, fx* pc[0] * pc[1] * inv_z2, -fx - fx * pc[0] * pc[0] * inv_z2, fx* pc[1] * inv_z,
			0, -fy * inv_z, fy* pc[1] * inv_z2, fy + fy * pc[1] * pc[1] * inv_z2, -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;
		J.block<2, 6>(2 * i, 0) = Ji;
		e.block<2, 1>(2 * i, 0) = ei;
	}
}

void LevenbergMarquardtSingle::Hessian()
{
	H = J.transpose() * J;
	g = -J.transpose() * e;
}

double LevenbergMarquardtSingle::GetCost()
{
	Eigen::MatrixXd cost = e.transpose() * e;
	return cost(0, 0);
}

double LevenbergMarquardtSingle::F(SE3d p)
{
	Eigen::MatrixXd ex;
	ex.resize(2 * points2d.size(), 1);
	double fx = focal(0);
	double fy = focal(1);
	double cx = center(0);
	double cy = center(1);
	for (size_t i = 0; i < points2d.size(); i++)
	{
		Eigen::Vector3d pc = p * points3d[i];
		Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
		Eigen::Vector2d ei = points2d[i] - proj;
		ex.block<2, 1>(2 * i, 0) = ei;
	}
	Eigen::MatrixXd F = 0.5 * ex.transpose() * ex;
	return F(0, 0);
}

double LevenbergMarquardtSingle::L0L(Vector6d dx)
{
	Eigen::MatrixXd L = -dx.transpose() * J.transpose() * e - 0.5 * dx.transpose() * J.transpose() * J * dx;
	return L(0, 0);
}

void LevenbergMarquardtSingle::Solution(Eigen::Matrix<double, 7, 1>& qt, double& pixel_error, int& diverged)
{
	int k = 0;
	double nu = 2.0;
	Jacobian();
	Hessian();
	bool found = (g.lpNorm<Eigen::Infinity>() < epsilon1);
	Vector6d A;
	A = H.diagonal();
	double max_p = A.maxCoeff();
	double mu = tau * max_p;

	double time = 0;

	while (!found && k < max_iter)
	{
		Runtimer t;
		t.start();

		k = k + 1;
		Eigen::Matrix<double, 6, 6> G = H + mu * Eigen::Matrix<double, 6, 6>::Identity();
		Vector6d h_ = G.ldlt().solve(g);
		Vector6d h;

		double r_ratio = 0.9;
		double t_ratio = 1.28;

		h << h_(0) * r_ratio, h_(1) * r_ratio, h_(2) * r_ratio, h_(3) * t_ratio, h_(4) * t_ratio, h_(5) * t_ratio;  //best

		if (h.norm() <= epsilon2 * ((pose.matrix()).squaredNorm() + epsilon2))
		{
			found = true;
		}
		else
		{
			SE3d pose_new = SE3d::exp(h) * pose;
			double rho = (F(pose) - F(pose_new)) / L0L(h);
			if (rho > 0)
			{
				pose = pose_new;
				Jacobian();
				Hessian();

				found = (g.lpNorm<Eigen::Infinity>() < epsilon1);
				mu = 1.8 * mu * std::max<double>(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
				nu = 2.0;
			}
			else
			{
				mu = mu * nu;
				nu = 2 * nu;
			}
		}
		t.stop();
		if (found == true)
		{
			Matrix3d rotation_matrix = pose.matrix().block(0, 0, 3, 3);
			Eigen::Quaterniond q(rotation_matrix);
			Vector3d t = pose.matrix().block(0, 3, 3, 1);
			qt << q.w(), q.x(), q.y(), q.z(), t;
			pixel_error = GetCost();
		}
	}
	if (found == false)
	{
		//std::cout << endl << "Diverged" << endl << endl;
		Matrix3d rotation_matrix = p_ini.block(0, 0, 3, 3);
		Eigen::Quaterniond q(rotation_matrix);
		Vector3d t = p_ini.block(0, 3, 3, 1);
		qt << q.w(), q.x(), q.y(), q.z(), t;
		pixel_error = 0.0;
		diverged = 1;
	}
}