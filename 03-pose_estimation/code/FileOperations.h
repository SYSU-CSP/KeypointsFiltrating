#pragma once
#include "HeadPose.h"

class FileOperations
{
private:
	string readFileIntoString(const char* filename);

	void DeleteStr(char del, string& str);

	bool findPosition(string str, string key, string& choose);
	
	void SplitStr(vector<Vector2d>& pts, string str, int num_split);

public:
	FileOperations() {}
	~FileOperations() {}

	void GetPoints(const char* pth, vector<Vector2d>& pts);

	bool getAllFiles(string fileName, vector<string>& files);

	//Read Data
	void ReadPoints2d(string path, vector<Vector2d>& pt2);

	void ReadPoints3d(string path, vector<Vector3d>& pt3);

	void ReadMatrix3d(string path, Matrix3d& intrinsic);

	void ReadFormatPoints2d(const char* pth, vector<Vector2d>& pts);

	void ReadPoint7d(string path, Vector7d& pt7);

	void read_distcoeff(string distortion_path, Vector5d distcoeff);

	//Write Data
	void WritePoint2d(string path, vector<Vector2d> points2d);

	void WritePose(string path, Vector7d pose);

	void WriteError(string path, vector<Vector6d> error);

	void WriteScore(string path, vector<double> score);

	void WriteResidual(string path, double residual);

	void WriteErrorData(string path, vector<string> error_datas);

};
