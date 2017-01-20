#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <string>
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <time.h>

// ./bin/buildModel ./KinectJoints.yaml 1000 8
//rosrun gmm_training buildModel3D ./data/KinectJoints.yaml 10000 25 8

using namespace std;
using namespace cv;

// Create Rotation matrix from roll, pitch and yaw
cv::Mat rpy(double roll, double pitch, double yaw)
{
	cv::Mat R1 = (Mat_<float>(3, 3) << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
	cv::Mat R2 = (Mat_<float>(3, 3) << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));
	cv::Mat R3 = (Mat_<float>(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);

	return  R3*R2*R1;
}

// Generate random 2D camera projects of 3D points X
// X is N samples x 24 (3 coordinates x 8 joints)
cv::Mat generate2Dpoints(cv::Mat X, cv::Mat K, int N, bool pairs)
{
	cv::Mat P, T, Xh;
	cv::Mat bins, scalingMat, Xim, Xim_t, bins_t;
	cv::Mat output(N,30,CV_32FC1);
	cv::Mat sample(1,30,CV_32FC1);
	RNG rng;
	double roll = 0, pitch = 0, yaw = 0, tx = 0, ty = 0, tz = 0;
	int sampleNum=0;
	//#pragma omp parallel for
	for (int n = 0; n < N; n++)
	{
		if ((n % 2 == 0)||!pairs) // Use the same camera projection for two iterations at a time, to keep some sequential information so as to learn gamma
		{
			sampleNum = rng.uniform(0,X.rows-1); // draw random sample from training data
			
			roll = rng.uniform(-5.0*M_PI/180.0, 5.0*M_PI/180.0);
			pitch = rng.uniform(-5.0*M_PI/180.0, 5.0*M_PI/180.0);
			yaw = rng.uniform(-5.0*M_PI/180.0, 5.0*M_PI/180.0);

			tx = rng.uniform(-0.2, 0.2);
			ty = -0.25 + rng.uniform(-0.2, 0.2);
			tz = rng.uniform(1.5, 2.0);
		}
		else
		{
			roll = roll + rng.gaussian(5.0*M_PI/180);
			pitch = pitch + rng.gaussian(5.0*M_PI/180);
			yaw = yaw + rng.gaussian(5.0*M_PI/180);
			tx = tx + rng.gaussian(0.05);
			ty = ty + rng.gaussian(0.05);
			tz = tz + rng.gaussian(0.05);
			sampleNum++;
		}
		sample = X.row(sampleNum);

		cv::Mat R = rpy(pitch,yaw,roll);
		cv::Mat t = (Mat_<float>(3, 1) << tx, ty, tz);
			
		
		hconcat(R, t, T);
		
		P = K*T;
		
		// Project all joints
		for (int i = 0; i < 8; i++) 
		{
			hconcat(sample.colRange(Range(3*i,3*i+3)),cv::Mat::ones(1,1,CV_32F),Xh);
			bins = P*Xh.t();
			vconcat(bins.row(2),bins.row(2),scalingMat);
			divide(bins.rowRange(Range(0,2)),scalingMat,Xim);
			Xim_t = Xim.t();
			bins_t = (bins.row(2)).t();
			Xim_t.copyTo(output(Range(n,n+1),Range(3*i,3*i+2)));
			bins_t.copyTo(output(Range(n,n+1),Range(3*i+2,3*i+3)));
		}
		cv::Mat cam = (Mat_<float>(1, 6) << roll, pitch, yaw, tx, ty, tz);
		cam.copyTo(output(Range(n,n+1),Range(24,30)));
	}
	return output;
}

cv::Mat MLParameterEstimate(cv::Mat X, cv::Mat mean, std::vector<cv::Mat> cov)
{
	//X.convertTo(X,mean.type());
	int N = X.rows/2;
	int scaleVal = N/100;
	cv::Mat gamma((int)cov.size(),N/scaleVal,CV_64F);
	cv::Mat m1(X.cols,1,mean.type());
	cv::Mat m2(X.cols,1,mean.type());
	cv::Mat u_res(1,1,CV_32F);
	cv::Mat v_res(1,1,CV_32F);
	int d2 = 2*X.cols;
	cv::Mat si_inv;
	for (int i = 0; i < (int)cov.size(); i++)
	{
		si_inv = cov[i].inv();
		double u = 0, v = 0;
		for (int j = 0; j < X.rows; j+=2)
		{
			X.row(j).convertTo(m1,mean.type());
			m1 = m1-mean.row(i);
			X.row(j+1).convertTo(m2,mean.type());
			m2 = m2-mean.row(i);
			u_res = (m1*si_inv*m1.t() + m2*si_inv*m2.t());
			v_res = (m1*si_inv*m2.t() + m2*si_inv*m1.t());
			//cout << u_res << " " << v_res << std::flush;
			u = u + u_res.at<double>(0);
			v = v + v_res.at<double>(0);
			if (j%scaleVal == 0) // Save every 1000th estimates
			{
				double a = j/2*d2;
				double b = -v;
				double c = 2.0*u-a;
				double d = -v;
				std::complex<double> del0 = pow(b,2)-3.0*a*c;
				std::complex<double> del1 = 2.0*pow(b,3) - 9.0*a*b*c + 27.0*pow(a,2)*d;		
				std::complex<double> temp = (del1 + sqrt(pow(del1,2)-4.0*pow(del0,3)))/2.0;
				std::complex<double> C = std::pow(temp,1.0/3.0);
				std::complex<double> g = -1.0/(3.0*a)*(b + C + del0/C); 
				//cout << "row: " << i << " col: " << j << " val: " << temp << " " << g << std::endl;
				gamma.at<double>(i,j/2/scaleVal) = std::real(g);
			}
		}
	}
	return gamma;
}


int main( int argc, char** argv )
{
	//ros::init(argc, argv, "gmmTraining");
	try {
		cout << "Loading training data " << argv[1] << " ... " << std::flush;
		//Load Data
		string sampleFile  = argv[1];
		
		Mat samples;
		FileStorage fs(sampleFile, FileStorage::READ);
		fs["Kinect"] >> samples;
		fs.release();
		cout << "Done." << std::endl;
		
		cout << "Generating " << atoi(argv[2]) << " 2D projections for training... " << std::flush;
	// Drone 
	//    cv::Mat K = (Mat_<float>(3, 3) << 205, 0, 158.3, 0, 207, 118.5, 0, 0, 1);
	//Kinect
		cv::Mat K  = (Mat_<float>(3, 3) << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0);
	//HDCam
		//~ cv::Mat K  = (Mat_<float>(3, 3) << 660.326889, 0.0, 318.705890, 0.0, 660.857176, 240.784699, 0.0, 0.0, 1.0);
	// Sign_language
		//cv::Mat K = (Mat_<float>(3,3) << 1550.0, 0.0, 130.0, 0.0, 1550.0, 105.0, 0.0, 0.0, 1.0);

		
		cv::Mat Xim = generate2Dpoints(samples, K, atoi(argv[2]),false);
		ofstream myfile; //Save generated samples data
		//myfile.open("./raw3D.txt");
		//myfile << Xim;
		//myfile.close();    
		//cout << "Done." << std::endl;
		
		// Xim handL elbowL shoulderL head    neck     shoulderR elbowR   handR      r  p  y  tx ty tz
		//	   0 1   2 3     4 5      6 7      8 9      10 11    12 13    14 15
		//	   0 1 2 3 4 5  6 7 8    9 10 11 12 13 14  15 16 17  18 19 20  21 22 23  24 25 26 27 28 29
	   // cout << Xim << std::endl;
	   
		time_t begin, end; 
		time(&begin);
		cout << "Training GMMs with " << atoi(argv[3]) << " clusters in " << atoi(argv[4]) << " dimensional space... " << std::endl;
		// Train GMM
		cv::Mat likelihoods1, labels1, probs1, likelihoods2, labels2, probs2;
		
		cv::Ptr<cv::ml::EM> gmm1 = cv::ml::EM::create();
		cv::Ptr<cv::ml::EM> gmm2 = cv::ml::EM::create();
		
		gmm1->setClustersNumber(atoi(argv[3]));
		gmm2->setClustersNumber(atoi(argv[3]));
		gmm1->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
		gmm2->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
		gmm1->setTermCriteria(cv::TermCriteria(300, 0.1, CV_TERMCRIT_ITER|CV_TERMCRIT_EPS));
		gmm2->setTermCriteria(cv::TermCriteria(300, 0.1, CV_TERMCRIT_ITER|CV_TERMCRIT_EPS));
		
		//#pragma omp parallel for	
		cv::Mat reverseCols;
		hconcat(Xim.colRange(21,24), Xim.colRange(18,21), reverseCols);
		hconcat(reverseCols,Xim.colRange(15,18),reverseCols);
		hconcat(reverseCols,Xim.colRange(9,15),reverseCols);//handR elbowR shoulderR head neck cam
		hconcat(reverseCols,Xim.colRange(24,30),reverseCols);
		
		cv::Mat cols1;
		hconcat(Xim.colRange(0,15), Xim.colRange(24,30), cols1);
				
		PCA pca1;
		PCA pca2;
		#pragma omp parallel for	
		for (int i = 0; i < 2; i++)
		{
			if (i == 0)
			{
				cout << "GMM 1..." << std::flush;
				pca1 = PCA(cols1,cv::Mat(),CV_PCA_DATA_AS_ROW,atoi(argv[4]));
				gmm1->trainEM(pca1.project(cols1), likelihoods1, labels1, probs1);
				cout << " Done. ";
			}
			else
			{
				cout << "GMM 2..." << std::flush;
				pca2 = PCA(reverseCols,cv::Mat(),CV_PCA_DATA_AS_ROW,atoi(argv[4]));
				gmm2->trainEM(pca2.project(reverseCols), likelihoods2, labels2, probs2);
				cout << " Done. ";
			}
		}
	
		time(&end);
		cout << "Time elapsed: " << difftime(end, begin) << " seconds" << endl;
	  
		// Learn gamma using Maximum likelihood
		time(&begin);
		cout << std::endl << "ML parameter estimation using " << atoi(argv[2]) << " points." << std::endl;
		Xim = generate2Dpoints(samples, K, atoi(argv[2]),true); // generate new points
		hconcat(Xim.colRange(21,24), Xim.colRange(18,21), reverseCols);
		hconcat(reverseCols,Xim.colRange(15,18),reverseCols);
		hconcat(reverseCols,Xim.colRange(9,15),reverseCols);
		hconcat(reverseCols,Xim.colRange(24,30),reverseCols);
		hconcat(Xim.colRange(0,15), Xim.colRange(24,30), cols1);
		
		cv::Mat g1,g2;
		std::vector<cv::Mat> covs1;
		gmm1->getCovs(covs1);
		std::vector<cv::Mat> covs2;
		gmm2->getCovs(covs2);//<std::vector<cv::Mat> >("covs");
		
		#pragma omp parallel for	
		for (int i = 0; i < 2; i++)
		{
			if (i == 0)
			{
				cout << "Gamma 1..." << std::flush;
				g1 = MLParameterEstimate(pca1.project(cols1), gmm1->getMeans(), covs1);
				cout << " Done. ";
			}
			else
			{
				cout << "Gamma 2..." << std::flush;
				g2 = MLParameterEstimate(pca2.project(reverseCols), gmm2->getMeans(), covs2);
				cout << " Done. ";
			}
		}
		time(&end);
		cout << "Time elapsed: " << difftime(end, begin) << " seconds" << endl;
	  
		cout << "Saving GMM parameters... " << std::flush;  
		
		//myfile.open("/tmp/GammaL3D_PCA.txt");
		//myfile << g1;
		//myfile.close();
		
		//myfile.open("/tmp/MeansL3D_PCA.txt");
		//myfile << gmm1.get<cv::Mat>("means");
		//myfile.close();
		
		//myfile.open ("/tmp/CovsL3D_PCA.txt");
		
		//for (int i = 0; i < (int)covs1.size(); i++)
		//{
			//myfile << covs1[i] << std::endl;
		//}
		//myfile.close();
		
		//myfile.open ("/tmp/WeightsL3D_PCA.txt");
		//myfile << gmm1.get<cv::Mat>("weights");
		//myfile.close();
		
		//myfile.open("/tmp/GammaR3D.txt");
		//myfile << g2;
		//myfile.close();
		
		//myfile.open ("/tmp/MeansR3D_PCA.txt");
		//myfile << gmm2.get<cv::Mat>("means");
		//myfile.close();
		
		//myfile.open ("/tmp/CovsR3D_PCA.txt");

		//for (int i = 0; i < (int)covs2.size(); i++)
		//{
			//myfile << covs2[i] << std::endl;
		//}
		//myfile.close();
		
		//myfile.open ("/tmp/WeightsR3D_PCA.txt");
		//myfile << gmm2.get<cv::Mat>("weights");
		//myfile.close();
		
		//myfile.open("/tmp/HL_PCA.txt");
		//myfile << pca1.eigenvectors;
		//myfile.close();    
		
		//myfile.open("/tmp/HR_PCA.txt");
		//myfile << pca2.eigenvectors;
		//myfile.close();    
		
		//myfile.open("/tmp/ML_PCA.txt");
		//myfile << pca1.mean;
		//myfile.close();    
		
		//myfile.open("/tmp/MR_PCA.txt");
		//myfile << pca2.mean;
		//myfile.close();    
		
		char fname[50];
		sprintf(fname,"./data13D_PCA_%d_%d_%d.yml",atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
		FileStorage fs1(fname, FileStorage::WRITE);
		sprintf(fname,"./data23D_PCA_%d_%d_%d.yml",atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
		FileStorage fs2(fname, FileStorage::WRITE);
		
		fs1 << "means" << gmm1->getMeans();
		fs2 << "means" << gmm2->getMeans();
		
		cv::Mat covs1tot = covs1[0];	
		cv::Mat covs2tot = covs2[0];	
		for (int i = 1; i < (int)covs1.size(); i++)
		{
			vconcat(covs1tot,covs1[i],covs1tot);
			vconcat(covs2tot,covs2[i],covs2tot);
		}
		fs1 << "covs" << covs1tot;
		fs2 << "covs" << covs2tot;
		
		fs1 << "weights" << gmm1->getWeights();
		fs2 << "weights" << gmm2->getWeights();
		
		fs1 << "pca_proj" << pca1.eigenvectors;
		fs2 << "pca_proj" << pca2.eigenvectors;
		
		fs1 << "pca_mean" << pca1.mean;
		fs2 << "pca_mean" << pca2.mean;
		
		fs1 << "gamma" << g1.col(g1.cols-1);
		fs2 << "gamma" << g2.col(g2.cols-1);
		
		fs1.release();
		fs2.release();
				
		cout << "Done." << std::endl; 
		
		return 0;
	}
	catch (exception& e)
	{
		cout << e.what() << std::endl << "Required arguments: data_file.yaml num_training_points num_clusters num_dimensions" << std::endl;
	}
}
