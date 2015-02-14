//
//  PCL_Interface.h
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 14..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#ifndef __sfm__PCL_Interface__
#define __sfm__PCL_Interface__

#include <stdio.h>
#include <vector>

#include <opencv2/core/core.hpp>

void fileSave(const std::vector<cv::Point3d>& pointcloud,
               const std::vector<cv::Vec3b>& pointcloud_RGB,
               const std::string& name);

#endif /* defined(__sfm__PCL_Interface__) */
