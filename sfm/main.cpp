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
    
    system ("pwd");
    
    char *default_dir = ".";
    
    const char *image_dir = argv[1]? argv[1] : default_dir;
    
    open_imgs_dir(image_dir,images,images_names);
	
    if(images.size() == 0) {
		cerr << "can't get image files" << endl;
		return 1;
	}
    cerr << "images loaded: " << endl;
    for (int i=0; i<images_names.size(); i++)
        cerr << i << ": " << images_names[i] << endl;
	
	cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names,string(image_dir));

    distance->use_rich_features = true;
    //
    // perform SFM
    //
    distance->RecoverDepthFromImages();
    
    // save the result
    // result3D.txt : 3D coordinates + RGB
    // features.txt : for each X_3D, its list of (image index, keypoint location, feature vector)
    // cameras.txt   : camera matrices
    fileSave(distance->getPointCloud(), distance->getPointCloudRGB(), "result3D.txt");

    return 0;
}

