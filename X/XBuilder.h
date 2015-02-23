//
//  XBuilder.h
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#ifndef __sfm__XBuilder__
#define __sfm__XBuilder__

#include <unistd.h>
#include <sys/param.h>

#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace cv;


struct XBuilder {
  
    vector<Mat> images;
    vector<string> image_names;
    
    // 3D reconstructed.
    std::vector<cv::Point3d> pointcloud;
    std::vector<cv::Vec3b>   pointcloud_RGB;
    
    void open_imgs_dir(string image_dir);
    void fileSave(const std::string name);

    void sfm();
};
#endif /* defined(__sfm__XBuilder__) */
