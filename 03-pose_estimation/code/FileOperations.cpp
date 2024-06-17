#pragma once
#include "FileOperations.h"

string FileOperations::readFileIntoString(const char* filename)

{
	ifstream in(filename);
	ostringstream buf;
	char ch;
	while (buf && in.get(ch))
	{
		buf.put(ch);
	}
	return buf.str();
}

void FileOperations::DeleteStr(char del, string& str)
{
	string::iterator it;
	for (it = str.begin(); it < str.end(); it++)
	{
		if (*it == del)
		{
			str.erase(it);
			it--;
		}
	}
}

bool FileOperations::findPosition(string str, string key, string& choose)
{
	int pos;
	if ((pos = str.find(key)) != string::npos)
	{
		string str_t;
		choose = str.substr(pos + key.size() + 2);
		return true;
	}
	else
	{
		cout << "Find Position No Find!" << endl;
		return false;
	}
}

void FileOperations::SplitStr(vector<Vector2d>& pts, string str, int num_split)
{
	char* s_input = (char*)str.c_str();
	const char* split = ",";
	char* p = NULL;
	char* ptr = NULL;
	p = strtok_s(s_input, split, &ptr);

	double temp[2] = { 0.0 };
	for (size_t i = 0; i < num_split; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{
			temp[j] = atof(p);
			p = strtok_s(NULL, split, &ptr);
		}
		Vector2d vt(temp[0], temp[1]);
		pts.push_back(vt);
	}
}

void FileOperations::GetPoints(const char* pth, vector<Vector2d>& pts)
{
	string str;
	str = readFileIntoString(pth);

	string key = "\"joints\"";
	string choose;
	findPosition(str, key, choose);
	DeleteStr('[', choose);
	DeleteStr(']', choose);
	DeleteStr('}', choose);
	DeleteStr(' ', choose);
	DeleteStr('\n', choose);
	SplitStr(pts, choose, 11);
}

void FileOperations::ReadFormatPoints2d(const char* pth, vector<Vector2d>& pts)
{
	string str;
	str = readFileIntoString(pth);
	string key = "[";
	string choose;
	findPosition(str, key, choose);
	DeleteStr('[', choose);
	DeleteStr(']', choose);
	SplitStr(pts, choose, 11);
}

void FileOperations::ReadPoints2d(string path, vector<Vector2d>& pt2)
{
	pt2.clear();
	Vector2d temp;

	ifstream fin(path, ios::in);
	if (!fin.is_open())
	{
		cout << "Read Points3d Path ERROR!" << '\t' << path << endl;
	}

	while (!fin.eof())
	{
		for (int j = 0; j < 2; j++)
		{

			fin >> temp[j]; 
		}
		pt2.push_back(temp);
	}
	pt2.pop_back();
	fin.close();
}

void FileOperations::ReadPoints3d(string path, vector<Vector3d>& pt3)
{
	pt3.clear();
	Vector3d temp;

	ifstream fin(path, ios::in);
	if (!fin.is_open())
	{
		cout << "Read Points3d Path ERROR!" << '\t' << path << endl;
	}

	while (!fin.eof())
	{
		for (int j = 0; j < 3; j++)
		{
			if (j == 0) { fin >> temp[0]; }
			if (j == 1) { fin >> temp[1]; }
			if (j == 2) { fin >> temp[2]; }
		}
		pt3.push_back(temp);
	}
	pt3.pop_back();
	fin.close();
}

void FileOperations::ReadMatrix3d(string path, Matrix3d& intrinsic)
{
	ifstream fin(path, ios::in);
	if (!fin.is_open())
	{
		cout << "ReadMatrix3d Path ERROR!" << '\t' << path << endl;
	}

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			fin >> intrinsic(i, j);
		}
	}
	fin.close();
}

bool FileOperations::getAllFiles(string fileName, vector<string>& files)
{
	int Num = 0;
	_finddata_t fileInfo;
	long long handle = _findfirst(fileName.c_str(), &fileInfo);

	if (handle == -1L)
	{
		cerr << "failed to transfer files" << endl;
		return false;
	}

	do
	{
		Num++;
		files.push_back(fileInfo.name);
	} while (_findnext(handle, &fileInfo) == 0);

	return true;
}

void FileOperations::WritePoint2d(string path, vector<Vector2d> points2d)
{
	ofstream fout;
	fout.open(path, ios::out);
	if (!fout.is_open())
	{
		cout << "WritePoint2d ERROR!" << '\t' << path << endl;
	}
	for (size_t i = 0; i < points2d.size(); i++)
	{
		fout << setiosflags(ios::fixed) << setprecision(8) << setiosflags(ios::left);
		fout << points2d[i](0) << '\t' << points2d[i](1) << '\t' << endl;
	}
	fout << endl;
	fout.close();
}

void FileOperations::WritePose(string path, Vector7d pose)
{
	ofstream fout;
	fout.open(path, ios::out);
	if (!fout.is_open())
	{
		cout << "WritePose ERROR!" << '\t' << path << endl;
	}

	for (size_t i = 0; i < 7; i++)
	{
		fout << setiosflags(ios::fixed) << setprecision(8) << setiosflags(ios::left);
		fout << pose(i) << '\t';
	}
	fout << endl;
}

void FileOperations::ReadPoint7d(string path, Vector7d& pt7)
{

	ifstream fin(path, ios::in);
	if (!fin.is_open())
	{
		cout << "ReadPoints7d ERROR!" << '\t' << path << endl;
	}

	for (int j = 0; j < 7; j++)
	{
		fin >> pt7[j];
	}

	fin.close();
}

void FileOperations::read_distcoeff(string distortion_path, Vector5d distcoeff)
{
	ifstream fin(distortion_path, ios::in);
	if (!fin.is_open())
	{
		cout << "read distcoeff ERROR!" << '\t' << distortion_path << endl;
	}

	for (size_t i = 0; i < 1; i++)
	{
		for (size_t j = 0; j < 5; j++)
		{
			fin >> distcoeff(i, j);
		}
	}
	fin.close();
}

void FileOperations::WriteError(string path, vector<Vector6d> error)
{
	ofstream fout;
	fout.open(path, ios::out);
	if (!fout.is_open())
	{
		cout << "WritePose ERROR!" << '\t' << path << endl;
	}

	for (size_t i = 0; i < error.size(); i++)
	{
		fout << setiosflags(ios::fixed) << setprecision(8) << setiosflags(ios::left);
		fout << error[i](0) << '\t' << error[i](1) << '\t' << error[i](2) << '\t';
		fout << error[i](3) << '\t' << error[i](4) << '\t' << error[i](5) << endl;
	}
	fout << endl;
	fout.close();
}

void FileOperations::WriteScore(string path, vector<double> score)
{
	ofstream fout;
	fout.open(path, ios::out);
	if (!fout.is_open())
	{
		cout << "WritePose ERROR!" << '\t' << path << endl;
	}

	for (size_t i = 0; i < score.size(); i++)
	{

		fout << score[i] << endl;
	}
	fout << endl;
	fout.close();
}

void FileOperations::WriteResidual(string path, double residual)
{
	ofstream fout;
	fout.open(path, ios::out);
	if (!fout.is_open())
	{
		cout << "WritePose ERROR!" << '\t' << path << endl;
	}
	fout << residual << endl;
	fout << endl;
	fout.close();
}

void FileOperations::WriteErrorData(string path, vector<string> error_datas)
{
	ofstream fout;
	fout.open(path, ios::out);
	if (!fout.is_open())
	{
		cout << "WritePose ERROR!" << '\t' << path << endl;
	}

	for (size_t i = 0; i < error_datas.size(); i++)
	{
		fout << error_datas[i] << endl;
	}
	fout << endl;
	fout.close();
}