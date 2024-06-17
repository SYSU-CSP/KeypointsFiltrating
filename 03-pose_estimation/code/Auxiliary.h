#pragma once
#include "HeadPose.h"

class Auxiliary
{
public:
	Auxiliary() {}
	~Auxiliary() {}

	void CalculatePrecision(Vector7d gt, Vector7d ret, double& dist_pre, double& angl_pre);

	double PoseScoreCalculation(Vector7d ret, Vector7d gt);

	void FilterPoints(vector<Vector3d> pt2_sc, vector<Vector3d> pt3_all, vector<Vector2d>& pt2, vector<Vector3d>& pt3, double filter_r);
};
