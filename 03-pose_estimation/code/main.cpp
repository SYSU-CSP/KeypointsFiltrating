#include "Brown.h"
#include "HeadPose.h"
#include "Auxiliary.h"
#include "EPnPInterface.h"
#include "FileOperations.h"
#include "SingleCameraLMInterface.h"

// check angle(precision)!

static int error_num = 0;

void PoseEstimation(vector<Vector3d> pt3, vector<Vector2d> pt2, Matrix3d intrinsic, Vector7d& ret, double& pixel_error, int& diverged)
{
	InterfaceEPnP* EPnP = InterfaceEPnP::CreateEPnP();
	EPnP->AddObservation(pt3, pt2, intrinsic);
	EPnP->InitialPose();

	double pixel_residual_init = 0.0;

	Matrix4d p0 = EPnP->GetInitialPose(pixel_residual_init);

	LevenbergMarquardtSingle* camera = new LevenbergMarquardtSingle();
	camera->AddObservation(pt3, pt2, intrinsic, p0, pixel_residual_init);
	camera->Solution(ret, pixel_error, diverged);
}

int main(int argc, char* argv[])
{
#if 1
	if (argc != 6)
	{
		cerr << "number of arguments must be 6 -v20231127!" << endl;
		cerr << "help:" << endl;
		cerr << "0.program - for example ./name.exe" << endl;
		cerr << "1.target model - for example ./model.json" << endl;
		cerr << "2.camera intrinsic - for example ./intrinsic_3x3.json" << endl;
		cerr << "3.pixel feature point - for example ./pred/" << endl;
		cerr << "4.filter points confidence - for example 0.8" << endl;
		cerr << "5.accurate points number - for example 8" << endl;
		return -1;
	}

	string model_path = argv[1];
	string intrinsic_path = argv[2];
	string pts2d_pred_path = argv[3];
	double filter_r = stod(argv[4]);
	int acc_num = stoi(argv[5]);

	/*string model_path = "./01-model/model-18.json";
	string intrinsic_path = "./02-intrinsic/intr.json";
	string pts2d_pred_path = "./03-pred-pts2/02-test/20221207-85-576x768/";
	string poses_result_path = "./04-result/02-test/20221207-85-576x768/";*/

	string pts2d_pred_files = pts2d_pred_path + "/*.json";

	string residual_path = "./residual/";
	string poses_result_path = "./pose_estimation/";

	if (_access(residual_path.c_str(), 0) == -1)
	{
		_mkdir(residual_path.c_str());
	}

	if (_access(poses_result_path.c_str(), 0) == -1)
	{
		_mkdir(poses_result_path.c_str());
	}

	Matrix3d intrinsic;
	vector<Vector3d> pt3_all;
	vector<string> pts2d_all_filename;

	FileOperations* file = new FileOperations();

	file->getAllFiles(pts2d_pred_files, pts2d_all_filename);
	file->ReadMatrix3d(intrinsic_path, intrinsic);
	file->ReadPoints3d(model_path, pt3_all);

	int pt_count = pt3_all.size();

	int num_pt2 = pts2d_all_filename.size();

	Auxiliary* auxi = new Auxiliary();

	int no_enough_count = 0;

	double pixel_error_sum = 0.0;
	double pixel_error_avg = 0.0;
	vector<double> pixel_error_vec;
	int diverged_all = 0;
	for (size_t i = 0; i < num_pt2; i++)
	{
		vector<Vector2d> pt2;
		vector<Vector3d> pt2_sc;
		vector<Vector3d> pt3;

		Vector7d ret;

		string pt2d_path = pts2d_pred_path + pts2d_all_filename[i];
		string pose_ret = poses_result_path + pts2d_all_filename[i];

		file->ReadPoints3d(pt2d_path, pt2_sc);

		auxi->FilterPoints(pt2_sc, pt3_all, pt2, pt3, filter_r);

		double pixel_error = 0.0;
		int diverged = 0;

		if (pt3.size() >= acc_num)
		{
			PoseEstimation(pt3, pt2, intrinsic, ret, pixel_error, diverged);
			diverged_all = diverged_all + diverged;
			pixel_error_vec.push_back(pixel_error);
			file->WritePose(pose_ret, ret);
		}

		else
		{
			no_enough_count++;
			//cout << pts2d_all_filename[i] << endl;
		}
	}
	pixel_error_sum = accumulate(pixel_error_vec.begin(), pixel_error_vec.end(), 0.0);
	pixel_error_avg = pixel_error_sum / (double)((num_pt2 - diverged_all) * pt_count);
	file->WriteScore(residual_path + "residual_all.json", pixel_error_vec);
	file->WriteResidual(residual_path + "residual.json", pixel_error_avg);

	cout << "Divergence Frequency:\t" << diverged_all << endl;
	cout << "Pixel Average Residual:\t" << pixel_error_avg << endl;
	std::cout << "No enough points:" << no_enough_count << endl;

#endif
	return 0;
}
