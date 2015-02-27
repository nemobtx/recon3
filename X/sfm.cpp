//
//  sfm.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"


void XBuilder::printReprojectionError()
{
    cerr << "XBuilder::printReprojectionError()" << endl;
//    for (int k=0; k<3; k++)
//        cerr << "R[" << k << "]:" << endl << this->R[k] << endl;
    
    double rms=0.;
    double mean=0.;
    int count=0;
    
    for (int i=0; i<x3d.size(); i++)
        {
        cv::Mat_<double> x3 (x3d[i].X);
        for (int k=0; k<x3d[i].ids.size(); k++)
            {
            if (x3d[i].ids[k] >= 0)
                {
                Mat_<double> p3 = R[k] * x3 + t[k];
                p3 /= p3(2);
                p3 = K * p3;
        
                cv::Point2f pt = this->imgKeypts[k][x3d[i].ids[k]].pt;

                double e1 = p3(0) - pt.x;
                double e2 = p3(1) - pt.y;

                //cerr << "R:" << R[k] << t[k] << "X:" << x3 << "pt" << pt << endl;

                double err2 = (e1*e1 + e2*e2);
                rms += err2;
                
                err2 = sqrt(err2);
                mean += err2;
                ++count;
                }
            }
        }
    rms = sqrt( rms / count );
    mean /= count;
    
    cerr << "! XBuilder:: Quality rms = " << rms << endl;
    cerr << "! XBuilder:: Quality mean L2 = " << mean << endl;
    cerr << "  ----" << endl;
}


void XBuilder::sfm()
{
    // 0. gray scale conversion
    
    // 1. keypoint + descriptor
    
    vector<pair<int, std::pair<int,int> > > pairs = KeyPoint_FMatrix_Matching ();
    
    // 2. 2-view reconstruction
    Two_View_Reconstruction(pairs);
    
    
    fileSave("result3D-2View.txt");

    // reconstruction for other views
}
