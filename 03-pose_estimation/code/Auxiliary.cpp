#pragma once
#include "Auxiliary.h"

void Auxiliary::CalculatePrecision(Vector7d pi, Vector7d pe, double& dist_pre, double& angl_pre)
{

	Eigen::Quaterniond qi(pi(0), pi(1), pi(2), pi(3));
	Eigen::Quaterniond qe(pe(0), pe(1), pe(2), pe(3));
	Eigen::Vector3d ri = qi.matrix().eulerAngles(0, 1, 2);
	Eigen::Vector3d re = qe.matrix().eulerAngles(0, 1, 2);

	Vector3d ti = pi.block<3, 1>(4, 0);
	Vector3d te = pe.block<3, 1>(4, 0);

	dist_pre = 100.0 * (te - ti).norm() / ti.norm();
	Matrix3d Ri = qi.matrix();
	Matrix3d Re = qe.matrix();

	double temp = acosf(((Re.transpose() * Ri).trace() - 1.0) / 2.0);

	if (isnan(temp)) { temp = (re - ri).norm(); }

	angl_pre = 100.0 * temp / ri.norm();

	cout << setiosflags(ios::fixed) << setprecision(8) << setiosflags(ios::left);

#if 0
	cout << "The translation vector of ground truth  :" << '\t' << ti.transpose() << endl;
	cout << "The translation vector of predict result:" << '\t' << tr.transpose() << endl << endl;

	cout << "The rotation vector of ground truth (angle)  :" << '\t' << 180 / M_PI * ri.transpose() << endl;
	cout << "The rotation vector of predict result (angle):" << '\t' << 180 / M_PI * rr.transpose() << endl << endl;
#endif
}

double Auxiliary::PoseScoreCalculation(Vector7d ret, Vector7d gt)
{

	Vector3d r_gt = gt.block(4, 0, 3, 1);
	Vector3d r_est = ret.block(4, 0, 3, 1);

	Vector4d q_gt = gt.block(0, 0, 4, 1);
	Vector4d q_est = ret.block(0, 0, 4, 1);

	double score_position = (r_gt - r_est).norm() / r_gt.norm();
	/*if (score_position < 0.002173)
	{
		score_position = 0.0;
	}*/

	double score_orientation = 2.0 * acos(q_gt.dot(q_est) / (q_gt.norm() * q_est.norm()));

	if (score_orientation > M_PI / 2.0)
	{
		q_est *= -1.0;
		score_orientation = 2.0 * acos(q_gt.dot(q_est) / (q_gt.norm() * q_est.norm()));
	}

	/*if (score_orientation < 0.169 * M_PI / 180.0)
	{
		score_orientation = 0.0;
	}*/

	double score_pose = score_orientation;

	/*if (score_pose > 0.0086)
	{
		cout << "score_pos:" << '\t' << score_position << endl;
		cout << "score_ori:" << '\t' << score_orientation << endl << endl;
	}*/

	return score_pose;
}

void Auxiliary::FilterPoints(vector<Vector3d> pt2_sc, vector<Vector3d> pt3_all, vector<Vector2d>& pt2, vector<Vector3d>& pt3, double filter_r)
{
	int num = pt2_sc.size();
	double threshold = filter_r;
	for (size_t i = 0; i < num; i++)
	{
		if (pt2_sc[i](2) > threshold)
		{
			pt2.push_back(Vector2d(pt2_sc[i](0), pt2_sc[i](1)));
			pt3.push_back(pt3_all[i]);
		}
	}
}
