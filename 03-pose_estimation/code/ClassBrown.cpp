#pragma once
#include "Brown.h"

pair<Matrix3d, Vector5d> BrownDistortion::GetInputData(vector<Vector2d> _poins2d, Matrix3d _intrinsic, Vector5d _distcoeff)
{
	pair<Matrix3d, Vector5d> IntrDist;
	poins2d.clear();
	poins2d.assign(_poins2d.begin(), _poins2d.end());
	intrinsic = _intrinsic;
	distcoeff = _distcoeff;
	IntrDist.first = intrinsic;
	IntrDist.second = distcoeff;

	return IntrDist;
}

//void BrownDistortion::GetValue()
//{
//	fx = intrinsic(0, 0);
//	fy = intrinsic(1, 1);
//	cx = intrinsic(0, 2);
//	cy = intrinsic(1, 2);
//
//	k1 = distcoeff(0, 0);
//	k2 = distcoeff(0, 1);
//	p1 = distcoeff(0, 2);
//	p2 = distcoeff(0, 3);
//	k3 = distcoeff(0, 4);
//}

vector<Vector2d> BrownDistortion::Distortion()
{
	vector<Vector2d> poins2d_dist;
	for (size_t i = 0; i < poins2d.size(); i++)
	{
		Vector2d pt;
		double xCorrected = (poins2d[i](0) - cx) / fx;
		double yCorrected = (poins2d[i](1) - cy) / fy;
		double r2 = xCorrected * xCorrected + yCorrected * yCorrected;
		double Ra = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
		double delta_x = 2 * p1 * xCorrected * yCorrected + p2 * (r2 + 2 * xCorrected * xCorrected);
		double delta_y = p1 * (r2 + 2 * yCorrected * yCorrected) + 2 * p2 * xCorrected * yCorrected;
		double x = xCorrected * Ra + delta_x;
		double y = yCorrected * Ra + delta_y;
		x = x * fx + cx;
		y = y * fy + cy;
		pt << x, y;
		poins2d_dist.push_back(pt);
	}
	return poins2d_dist;
}

vector<Vector2d> BrownDistortion::Undistortion(pair<Matrix3d, Vector5d> IntrDist)
{
	double delta = 10;
	double eps = 1e-12;

	fx = IntrDist.first(0, 0);
	fy = IntrDist.first(1, 1);
	cx = IntrDist.first(0, 2);
	cy = IntrDist.first(1, 2);

	k1 = IntrDist.second(0, 0);
	k2 = IntrDist.second(0, 1);
	p1 = IntrDist.second(0, 2);
	p2 = IntrDist.second(0, 3);
	k3 = IntrDist.second(0, 4);
	
	vector<Vector2d> point_corrected;
	for (size_t i = 0; i < poins2d.size(); i++)
	{
		Vector2d ptc;
		double xDistortion = (poins2d[i](0) - cx) / fx;
		double yDistortion = (poins2d[i](1) - cy) / fy;
		//cout << "xDistortion: " << '\t' << poins2d[i](0) << '\t' << "yDistortion: " << '\t' << poins2d[i](1) << endl;
		double x0 = xDistortion;
		double y0 = yDistortion;
		
		while (true)
		{
			double r2 = xDistortion * xDistortion + yDistortion * yDistortion;
			double Radial = 1 / (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
			double deltaX = 2 * p1 * xDistortion * yDistortion + p2 * (r2 + 2 * xDistortion * xDistortion);
			double deltaY = p1 * (r2 + 2 * yDistortion * yDistortion) + 2 * p2 * xDistortion * yDistortion;
			double xCorrected = (x0 - deltaX) * Radial;
			double yCorrected = (y0 - deltaY) * Radial;

			delta = fabs(sqrt(xDistortion * xDistortion + yDistortion * yDistortion) - sqrt(xCorrected * xCorrected + yCorrected * yCorrected));
			//cout << delta << endl;
			//cout <<"calculating " << i << " th" << endl;
			xDistortion = xCorrected;
			yDistortion = yCorrected;

			if (delta < eps)
			{
				double xc = xCorrected * fx + cx;
				double yc = yCorrected * fy + cy;
				ptc << xc, yc;
				point_corrected.push_back(ptc);
				
				break;
			}
		}
	}
	return point_corrected;
}