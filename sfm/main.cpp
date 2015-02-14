/*****************************************************************************
*   ExploringSfMWithOpenCV
******************************************************************************
*   by Roy Shilkrot, 5th Dec 2012
*   http://www.morethantechnical.com/
******************************************************************************
*   Ch4 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#include <unistd.h>
#include <sys/param.h>
#include <iostream>
#include <string.h>

#include "Distance.h"
#include "MultiCameraPnP.h"
//#include "Visualization.h"
#include "file_Interface.h"

using namespace std;

std::vector<cv::Mat> images;
std::vector<std::string> images_names;












int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << "USAGE: " << argv[0] << " <path_to_images> " << endl;
        //return 0;
	}
	
	double downscale_factor = 1.0;
	if(argc >= 5)
		downscale_factor = atof(argv[4]);

    // change working directory
    //chdir ("../../../../../data");
    //system("pwd");
    
    char *default_dir = "."; ///Users/yndk/Dropbox/fountain/fountain_dense/urd";
    
    const char *image_dir = argv[1]? argv[1] : default_dir;
    
    open_imgs_dir(image_dir,images,images_names,downscale_factor);
	
    if(images.size() == 0) {
		cerr << "can't get image files" << endl;
		return 1;
	}
    cerr << "images loaded: " << endl;
    for (int i=0; i<images_names.size(); i++)
        cerr << i << ": " << images_names[i] << endl;
	
	cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names,string(image_dir));
//	if(argc < 3)
//		distance->use_rich_features = true;
//	else
//		distance->use_rich_features = (strcmp(argv[2], "RICH") == 0);
	
    distance->use_rich_features = true;
    // rich_feature means: PyramidFAST + ORB

//	if(argc < 4)
//		distance->use_gpu = (cv::gpu::getCudaEnabledDeviceCount() > 0);
//	else
//		distance->use_gpu = (strcmp(argv[3], "GPU") == 0);
//
    distance->use_gpu = false;
    
    //
    // perform SFM
    //
    distance->RecoverDepthFromImages();
    
	//get the scale of the result cloud using PCA
	double scale_cameras_down = 1.0;
	{
		vector<cv::Point3d> cld = distance->getPointCloud();
		if (cld.size()==0) cld = distance->getPointCloudBeforeBA();
		cv::Mat_<double> cldm((int)cld.size(),3);
		for(unsigned int i=0;i<cld.size();i++) {
			cldm.row(i)(0) = cld[i].x;
			cldm.row(i)(1) = cld[i].y;
			cldm.row(i)(2) = cld[i].z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);
		scale_cameras_down = pca.eigenvalues.at<double>(0) / 5.0;
		//if (scale_cameras_down > 1.0) {
		//	scale_cameras_down = 1.0/scale_cameras_down;
		//}
	}
	
    fileSave(distance->getPointCloud(), distance->getPointCloudRGB(), "result3D.txt");

    return 0;
}

