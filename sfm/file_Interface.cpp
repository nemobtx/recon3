//
//  PCL_Interface.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 14..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include "file_Interface.h"
using namespace std;


void fileSave(const std::vector<cv::Point3d>& pointcloud,
               const std::vector<cv::Vec3b>& pointcloud_RGB,
               const std::string& name)
{
    FILE *fp = fopen (name.c_str(), "w");
    if (fp==0) {
        cerr << "File open error: " << name << endl;
        exit (0);
    }
    
    double t = cv::getTickCount();
    fprintf(fp, "%lu\n", pointcloud.size());
    for (unsigned int i=0; i<pointcloud.size(); i++) {
        // get the RGB color value for the point
        cv::Vec3b rgbv(255,255,255);
        if (pointcloud_RGB.size() > i) {
            rgbv = pointcloud_RGB[i];
        }
        
        // check for erroneous coordinates (NaN, Inf, etc.)
        if (pointcloud[i].x != pointcloud[i].x ||
            pointcloud[i].y != pointcloud[i].y ||
            pointcloud[i].z != pointcloud[i].z ||
#ifndef WIN32
            isnan(pointcloud[i].x) ||
            isnan(pointcloud[i].y) ||
            isnan(pointcloud[i].z) ||
#else
            _isnan(pointcloud[i].x) ||
            _isnan(pointcloud[i].y) ||
            _isnan(pointcloud[i].z) ||
#endif
            //fabsf(pointcloud[i].x) > 10.0 ||
            //fabsf(pointcloud[i].y) > 10.0 ||
            //fabsf(pointcloud[i].z) > 10.0
            false
            )
            {
            continue;
            }
        
        // 3D coordinates
//        pclp.x = pointcloud[i].x;
//        pclp.y = pointcloud[i].y;
//        pclp.z = pointcloud[i].z;

        fprintf (fp, "%10.7f  %10.7f  %10.7f  %4d  %4d  %4d\n",
                 pointcloud[i].x, pointcloud[i].y, pointcloud[i].z,
                 rgbv[0], rgbv[1], rgbv[2]);
    }
    fclose (fp);
    
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    cout << "Done. (" << t <<"s)"<< endl;
    cout << "! file save to : " << name << endl;
}

// EOF //