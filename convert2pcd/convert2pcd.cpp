//
//  main.cpp
//  convert2pcd
//
//  Created by Yongduek Seo on 2015. 2. 14..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

const string defaultFile = "result3D.txt";

int main(int argc, const char * argv[])
{
    string outputFile = "result3D.pcd";
    string inputFile = argc==2? argv[1] : defaultFile;

    ifstream fs (inputFile);
    if (!fs.is_open()) {
        cerr << "! input file not found: " << inputFile << endl;
        cerr << "usage: " << argv[0] << " [result3D.txt]" << endl;
        exit (1);
    }
    cerr << "! input file: " << inputFile << endl;
    
    // pcl cloud
    pcl::PointCloud<pcl::PointXYZRGB> mycloud;
    
    int N; // the number of data points
    fs >> N;
    for (int i=0; i<N; i++) {
        float x, y, z;
        uint32_t r, g, b;
        fs >> x >> y >> z; //
        fs >> r >> g >> b; // rgb
        
        pcl::PointXYZRGB pclp;
        
        pclp.x = x; pclp.y = y; pclp.z = z;

        uint32_t rgb = (r << 16 | g << 8 | b);
        pclp.rgb = *reinterpret_cast<float*>(&rgb);

        mycloud.push_back (pclp);
    }
    fs.close();
    
    mycloud.width = (uint32_t) mycloud.points.size();
    mycloud.height = 1;
    
    pcl::PCDWriter pw;
    pw.write(outputFile, mycloud);
    cerr << "! file written to: " << outputFile << endl;
    return 0;
}
