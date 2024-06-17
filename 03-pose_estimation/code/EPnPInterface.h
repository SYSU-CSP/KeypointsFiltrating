#pragma once
#include "HeadPose.h"

#define EPNP_API _declspec(dllexport)

class EPNP_API InterfaceEPnP
{
public:
	static InterfaceEPnP * CreateEPnP();

	virtual void AddObservation(vector<Vector3d> &points3d, vector<Vector2d> &points2d, Matrix3d &K) = 0;

	virtual void InitialPose() = 0;

	virtual Matrix4d GetInitialPose(double& residual) = 0;

};
