#pragma once
#include "EPnPInterface.h"

class EPNP :public InterfaceEPnP
{
private:
	struct DistPattern
	{
		int a;
		int b;
	};
	MatrixXd reference_3d_points_;
	MatrixXd reference_2d_points_;
	MatrixXd reference_3d_points_camera_coord_;
	MatrixXd control_3d_points_;
	MatrixXd control_3d_points_camera_coord_;
	MatrixXd bary_centric_coord_;
	Matrix3d poseR0;
	Vector3d poseT0;

	double epsilon1;
	double epsilon2;
	double tau;
	int max_iter;
	bool is_out;

	Eigen::Matrix<double, 6, 1> e;
	Eigen::MatrixXd J;
	Eigen::Matrix4d H;
	Eigen::Vector4d g;
	Vector3d result;

	double residual_0;

	int reference_points_count_;
	double fu_, fv_, uc_, vc_;

	void ChooseControlPoints();

	void CalculateBaryCenterCoord();

	void CalculateCoeffMatrix(MatrixXd &M);

	void CalculateBetas(const MatrixXd &U, MatrixXd &L);

	void CalculateRho(VectorXd &rho);

	void FindBetasApproxl(const MatrixXd L, const VectorXd &rho, double *betas);

	void Jacobian(const MatrixXd L, double betas[4]);

	void Hessian();

	double GetCost();

	double F(Eigen::Matrix<double, 6, 1> x);

	double LOL(Vector4d dx);

	Eigen::Matrix<double, 6, 1> Residuals(const MatrixXd U, double betas[4]);

	void GaussNewton(const MatrixXd &U, const MatrixXd &L, double betas[4]);

	void CalculateControlPointsCameraCoord(const MatrixXd &U, double betas[4]);

	void CalculateReferencePointsCameraCoord();

	void SolveForSign();

	void EstimatePose();

	void CalculatePose(const MatrixXd &U, double betas[4]);

public:
	EPNP();

	virtual void AddObservation(vector<Vector3d> &points3d, vector<Vector2d> &points2d, Matrix3d &K);

	virtual void InitialPose();

	virtual Matrix4d GetInitialPose(double& residual);

	~EPNP() {};

};

EPNP::EPNP()
{
	// 目前该超参数更准确
	epsilon1 = 1e-2;
	epsilon2 = 1e-12;
	tau = 1e-2;
	max_iter = 1000;
	is_out = true;

#if 0  //原设定的超参数
	epsilon1 = 1e-6;
	epsilon2 = 1e-12;
	tau = 1e-3;
	max_iter = 500;
	is_out = true;
#endif
}

void EPNP::AddObservation(vector<Vector3d> &points3d, vector<Vector2d> &points2d, Matrix3d &K)
{
	reference_points_count_ = points3d.size();
	reference_3d_points_.resize(reference_points_count_, 3);
	reference_2d_points_.resize(reference_points_count_, 2);

	for (int i = 0; i < reference_points_count_; i++)
	{
		reference_3d_points_.block(i, 0, 1, 3) << points3d[i](0), points3d[i](1), points3d[i](2);
		reference_2d_points_.block(i, 0, 1, 2) << points2d[i](0), points2d[i](1);
	}
	control_3d_points_ = MatrixXd::Zero(4, 3);
	control_3d_points_camera_coord_ = MatrixXd::Zero(4, 3);
	bary_centric_coord_ = MatrixXd::Zero(reference_points_count_, 4);
	reference_3d_points_camera_coord_ = MatrixXd::Zero(reference_points_count_, 3);

	fu_ = K(0, 0);
	fv_ = K(1, 1);
	uc_ = K(0, 2);
	vc_ = K(1, 2);
};

void EPNP::ChooseControlPoints()
{
	double lambda;
	VectorXd eigvec;
	MatrixXd pointsSum = reference_3d_points_.colwise().sum();
	pointsSum = pointsSum / reference_points_count_;
	control_3d_points_.row(0) = pointsSum;

	MatrixXd centroidMat = pointsSum.replicate(reference_points_count_, 1);
	MatrixXd PW0 = reference_3d_points_ - centroidMat;
	MatrixXd PW0t = PW0;
	PW0t.transposeInPlace();
	MatrixXd PW0tPW0 = PW0t * PW0;

	SelfAdjointEigenSolver<MatrixXd> es(PW0tPW0);
	VectorXd eigenval = es.eigenvalues();
	VectorXd k = (eigenval / reference_points_count_).cwiseSqrt();

	int sign_value[3] = { 1, -1, -1 };
	for (int i = 2; i >= 0; i--)
	{
		lambda = k(i);
		eigvec = es.eigenvectors().col(i);
		control_3d_points_.row(3 - i) = control_3d_points_.row(0) + sign_value[i] * lambda * eigvec.transpose();
	}
}

void EPNP::CalculateBaryCenterCoord()
{
	Matrix3d CC(3, 3);

	for (int i = 0; i < 3; i++)
	{
		CC.row(i) = control_3d_points_.row(i + 1) - control_3d_points_.row(0);
	}
	CC.transposeInPlace();

	MatrixXd CC_inv = CC.inverse();
	MatrixXd pt_3d_diff_mat(1, 3);
	Vector3d pt_3d_diff_vec;
	double alpha;
	for (int i = 0; i < reference_points_count_; i++)
	{
		pt_3d_diff_mat = reference_3d_points_.row(i) - control_3d_points_.row(0);
		pt_3d_diff_vec = pt_3d_diff_mat.transpose();
		pt_3d_diff_vec = CC_inv * pt_3d_diff_vec;
		alpha = 1.0 - pt_3d_diff_vec.sum();
		bary_centric_coord_(i, 0) = alpha;
		bary_centric_coord_.block(i, 1, 1, 3) = pt_3d_diff_vec.transpose();
	}
}

void EPNP::CalculateCoeffMatrix(MatrixXd &M)
{
	double uci, vci, barycentric;
	for (int i = 0; i < reference_points_count_; i++)
	{
		uci = uc_ - reference_2d_points_(i, 0);
		vci = vc_ - reference_2d_points_(i, 1);
		for (int j = 0; j < 4; j++)
		{
			barycentric = bary_centric_coord_(i, j);
			M(2 * i, 3 * j) = barycentric * fu_;
			M(2 * i, 3 * j + 1) = 0;
			M(2 * i, 3 * j + 2) = barycentric * uci;
			M(2 * i + 1, 3 * j) = 0;
			M(2 * i + 1, 3 * j + 1) = barycentric * fv_;
			M(2 * i + 1, 3 * j + 2) = barycentric * vci;
		}
	}
}

void EPNP::CalculateBetas(const MatrixXd &U, MatrixXd &L)
{
	MatrixXd V = U.block(0, 0, 12, 4);
	MatrixXd DiffMat = MatrixXd::Zero(18, 4);
	DistPattern diff_pattern[6] = { {0, 1},{0, 2},{0, 3},{1, 2},{1, 3},{2, 3} };

	for (int i = 0; i < 6; i++)
	{
		DiffMat.block(3 * i, 0, 3, 4) = V.block(3 * diff_pattern[i].a, 0, 3, 4) - V.block(3 * diff_pattern[i].b, 0, 3, 4);
	}

	Vector3d v1, v2, v3, v4;
	for (int i = 0; i < 6; i++)
	{
		v1 = DiffMat.block(3 * i, 0, 3, 1);
		v2 = DiffMat.block(3 * i, 1, 3, 1);
		v3 = DiffMat.block(3 * i, 2, 3, 1);
		v4 = DiffMat.block(3 * i, 3, 3, 1);

		L.block(i, 0, 1, 10) << v1.dot(v1), 2 * v1.dot(v2), v2.dot(v2), 2 * v1.dot(v3), 2 * v2.dot(v3),
			                    v3.dot(v3), 2 * v1.dot(v4), 2 * v2.dot(v4), 2 * v3.dot(v4), v4.dot(v4);
	}
}

void EPNP::CalculateRho(VectorXd &rho)
{
	Vector3d control_point_a, control_point_b, control_point_diff;
	DistPattern diff_pattern[6] = { {0, 1},{0, 2},{0, 3},{1, 2},{1, 3},{2, 3} };

	for (int i = 0; i < 6; i++)
	{
		control_point_a << control_3d_points_(diff_pattern[i].a, 0), control_3d_points_(diff_pattern[i].a, 1), control_3d_points_(diff_pattern[i].a, 2);
		control_point_b << control_3d_points_(diff_pattern[i].b, 0), control_3d_points_(diff_pattern[i].b, 1), control_3d_points_(diff_pattern[i].b, 2);
		control_point_diff = control_point_a - control_point_b;
		rho(i) = control_point_diff.dot(control_point_diff);
	}
}

void EPNP::FindBetasApproxl(const MatrixXd L, const VectorXd &rho, double *betas)
{
	MatrixXd Lm(6, 4);
	Lm.block(0, 0, 6, 2) = L.block(0, 0, 6, 2);
	Lm.block(0, 2, 6, 1) = L.block(0, 3, 6, 1);
	Lm.block(0, 3, 6, 1) = L.block(0, 6, 6, 1);

	VectorXd B = Lm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rho);

	if (B(0) < 0)
	{
		betas[0] = sqrt(-B(0));
		betas[1] = -B(1) / betas[0];
		betas[2] = -B(2) / betas[0];
		betas[3] = -B(3) / betas[0];
	}
	else
	{
		betas[0] = sqrt(B(0));
		betas[1] = B(1) / betas[0];
		betas[2] = B(2) / betas[0];
		betas[3] = B(3) / betas[0];
	}
}

void EPNP::Jacobian(const MatrixXd L, double betas[4])
{
	MatrixXd L2J(10, 4);
	L2J << 2 * betas[0],         0,          0,          0,
		   betas[1],      betas[0],          0,          0,
		   0,         2 * betas[1],          0,          0,
		   betas[2],             0,   betas[0],          0,
		   0,             betas[2],   betas[1],          0,
		   0,                0,     2 * betas[2],        0,
		   betas[3],         0,        0,         betas[0],
		   0,         betas[3],        0,         betas[1],
		   0,             0,    betas[3],         betas[2],
		   0,             0,           0,     2 * betas[3];
	J = L * L2J;
}

void EPNP::Hessian()
{
	H = J.transpose() * J;
	g = -J.transpose() * e;
}

double EPNP::GetCost()
{
	Eigen::MatrixXd cost = e.transpose() * e;
	return cost(0, 0);
}

double EPNP::F(Eigen::Matrix<double, 6, 1> x)
{
	MatrixXd F = 0.5 * x.transpose() * x;
	return F(0, 0);
}

double EPNP::LOL(Vector4d dx)
{
	MatrixXd L = -dx.transpose() * J.transpose() * e - 0.5 * dx.transpose() * J.transpose() * J * dx;
	return L(0, 0);
}

Eigen::Matrix<double, 6, 1> EPNP::Residuals(const MatrixXd U, double betas[4])
{
	Eigen::Matrix<double, 6, 1> re;    //for F
	MatrixXd V = U.block(0, 0, 12, 4);
	DistPattern diff_pattern[6] = { {0, 1},{0, 2},{0, 3},{1, 2},{1, 3},{2, 3} };
	VectorXd CC(12, 1);
	Vector3d Ca, Cb;
	MatrixXd Wa, Wb;
	Vector3d Vwa, Vwb;

	CC = betas[0] * V.block(0, 0, 12, 1) + betas[1] * V.block(0, 1, 12, 1) + betas[2] * V.block(0, 2, 12, 1) + betas[3] * V.block(0, 3, 12, 1);

	for (int i = 0; i < 6; i++)
	{
		Ca = CC.block(3 * diff_pattern[i].a, 0, 3, 1);
		Cb = CC.block(3 * diff_pattern[i].b, 0, 3, 1);
		Wa = control_3d_points_.block(diff_pattern[i].a, 0, 1, 3);
		Wb = control_3d_points_.block(diff_pattern[i].b, 0, 1, 3);

		Ca = Ca - Cb;
		Cb = Ca;
		double d1 = Ca.dot(Cb);

		Wa = Wa - Wb;
		Wa.transposeInPlace();
		Vwa = Wa;
		Vwb = Vwa;
		double d2 = Vwa.dot(Vwb);

		e(i) = d1 - d2;
	}
	re = e;
	return re;
}

void EPNP::GaussNewton(const MatrixXd &U, const MatrixXd &L, double betas[4])
{
	Vector4d Vb;
	Vb << betas[0], betas[1], betas[2], betas[3];

	int k = 0;
	double nu = 2.0;
	Jacobian(L, betas);
	Residuals(U, betas);
	Hessian();
	bool found = (g.lpNorm<Eigen::Infinity>() < epsilon1);

	Vector4d A;
	A = H.diagonal();
	double max_p = A.maxCoeff();
	double mu = tau * max_p;

	while (!found && k < max_iter)
	{
		k++;
		Matrix4d G = H + mu * Eigen::Matrix4d::Identity();
		Vector4d h = G.ldlt().solve(g); 
		if (h.norm() <= epsilon2 * (Vb.squaredNorm() + epsilon2))
		{
			found = true;
		}
		else
		{
			Vector4d Vb_new = h + Vb;
			Eigen::Matrix<double, 6, 1> x = Residuals(U, betas);
			betas[0] = Vb_new(0);
			betas[1] = Vb_new(1);
			betas[2] = Vb_new(2);
			betas[3] = Vb_new(3);
			Eigen::Matrix<double, 6, 1> x_new = Residuals(U, betas);

			double rho = (F(x) - F(x_new)) / LOL(h);
			if (rho > 0)
			{
				Vb = Vb_new;
				Jacobian(L, betas);
				Residuals(U, betas);
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
		if (is_out)
		{
			//std::cout << endl;
			//std::cout << "Iter: " << std::left << std::setw(3) << k << " Result: " << std::endl;
			//std::cout << std::endl << Vb.transpose() << endl << endl;
			//std::cout << " step: " << std::left << std::setw(14) << h.norm() << " cost: " << std::left << std::setw(14) << GetCost() << std::endl;
			residual_0 = GetCost();
		}
	}
	if (found == false)
	{
		//std::cout << endl << "EPnP Divergent" << endl << endl;
		residual_0 = 0.0;
	}
	/*const int iterations_number = 5;
	Eigen::Vector4d Vb;
	Eigen::Vector4d jacobian_res;

	Vb << betas[0], betas[1], betas[2], betas[3];
	for (int i = 0; i < iterations_number; i++)
	{
		Jacobian(L, betas);
		Residuals(U, betas);
		Hessian();

		Vb = Vb + H.inverse() * g;

		betas[0] = Vb(0);
		betas[1] = Vb(1);
		betas[2] = Vb(2);
		betas[3] = Vb(3);
	}*/
}

void EPNP::CalculateControlPointsCameraCoord(const MatrixXd &U, double betas[4])
{
	MatrixXd V = U.block(0, 0, 12, 4);
	MatrixXd control_3d_points_cam_coord_vec(12, 1);
	control_3d_points_cam_coord_vec = betas[0] * V.block(0, 0, 12, 1) + betas[1] * V.block(0, 1, 12, 1)
		                            + betas[2] * V.block(0, 2, 12, 1) + betas[3] * V.block(0, 3, 12, 1);
	for (int i = 0; i < 4; i++)
	{
		control_3d_points_camera_coord_.block(i, 0, 1, 3) << control_3d_points_cam_coord_vec(3 * i),
			                                                 control_3d_points_cam_coord_vec(3 * i + 1),
		                                                     control_3d_points_cam_coord_vec(3 * i + 2);
	}
}

void EPNP::CalculateReferencePointsCameraCoord()
{
	reference_3d_points_camera_coord_ = bary_centric_coord_ * control_3d_points_camera_coord_;
}

void EPNP::SolveForSign()
{
	if (reference_3d_points_camera_coord_(0, 2) < 0)
	{
		control_3d_points_camera_coord_ = -1 * control_3d_points_camera_coord_;
		reference_3d_points_camera_coord_ = -1 * reference_3d_points_camera_coord_;
	}
}

void EPNP::EstimatePose()
{
	MatrixXd pointsSumA = reference_3d_points_.colwise().sum();
	pointsSumA = pointsSumA / reference_points_count_;
	Vector3d P0w = pointsSumA.transpose();

	MatrixXd centroidMA = pointsSumA.replicate(reference_points_count_, 1);
	MatrixXd A = reference_3d_points_ - centroidMA;

	MatrixXd pointsSumB = reference_3d_points_camera_coord_.colwise().sum();
	pointsSumB = pointsSumB / reference_points_count_;
	Vector3d P0c = pointsSumB.transpose();

	MatrixXd centroidMB = pointsSumB.replicate(reference_points_count_, 1);
	MatrixXd B = reference_3d_points_camera_coord_ - centroidMB;

	Matrix3d H = B.transpose() * A;

	JacobiSVD<MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();

	poseR0 = U * V.transpose();
	double detR = poseR0.determinant();

	if (detR < 0)
	{
		poseR0(2, 0) = -poseR0(2, 0);
		poseR0(2, 1) = -poseR0(2, 1);
		poseR0(2, 2) = -poseR0(2, 2);
	}
	poseT0 = P0c - poseR0 * P0w;

}

void EPNP::CalculatePose(const MatrixXd &U, double betas[4])
{
	CalculateControlPointsCameraCoord(U, betas);
	CalculateReferencePointsCameraCoord();
	SolveForSign();
	EstimatePose();
}

void EPNP::InitialPose()
{
	ChooseControlPoints();

	CalculateBaryCenterCoord();

	MatrixXd M(2 * reference_points_count_, 12);

	M = MatrixXd::Zero(2 * reference_points_count_, 12);

	CalculateCoeffMatrix(M);

	MatrixXd MTM = M.transpose() * M;

	SelfAdjointEigenSolver<MatrixXd> es(MTM);

	VectorXd eigenval = es.eigenvalues();

	MatrixXd eigenvec = es.eigenvectors();

	eigenvec.block(0, 2, 12, 1) = -eigenvec.block(0, 2, 12, 1);

	eigenvec.block(0, 3, 12, 1) = -eigenvec.block(0, 3, 12, 1);

	MatrixXd L = MatrixXd::Zero(6, 10);

	VectorXd rho(6, 1);

	CalculateBetas(eigenvec, L);

	CalculateRho(rho);

	double betas[4][4];

	FindBetasApproxl(L, rho, betas[1]);

	GaussNewton(eigenvec, L, betas[1]);

	CalculatePose(eigenvec, betas[1]);
}

Matrix4d EPNP::GetInitialPose(double& residual)
{
	residual = residual_0;
	Matrix4d Qt;
	Qt << poseR0, poseT0, 0, 0, 0, 1;

	return Qt;
}

InterfaceEPnP * InterfaceEPnP::CreateEPnP()
{
	return new EPNP();
}
