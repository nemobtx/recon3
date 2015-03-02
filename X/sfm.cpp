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

void get2D3D (vector<Point2f>& q2, vector<Point3f>& q3, int i,
              set<int>& images_processed,
              MatchMap& matches_pairs,
              vector<P3D>& x3d,
              vector<vector<KeyPoint> >& imgKeypts)
{
    q2.clear();
    q3.clear();
    
    cerr << "get2d3d()" << endl;
    // find 2d-3D pairs from every possible image pairs
    vector<bool> usedX3(x3d.size(), false);
    for (set<int>::iterator it=images_processed.begin(); it!=images_processed.end(); it++)
    {
    int used_img_id = *it;
    std::pair<int,int> pair(used_img_id,i);
    vector<DMatch> matches = matches_pairs[pair];
    int count=0;
    for (int m=0; m<matches.size(); m++)
        {
        for (int i3=0; i3<x3d.size(); i3++)
            if (x3d[i3].ids[used_img_id] == matches[m].queryIdx
                && usedX3[i3]==false)
                {
                q3.push_back(x3d[i3].X);
                q2.push_back(imgKeypts[i][matches[m].trainIdx].pt);
                usedX3[i3]=true;
                count++;
                }
        }
    
    cerr << "\t\t>>  pair(" << used_img_id << "," << i << ") has " << count << " 2d3d matches" << endl;
    }
    cerr << " --- " << endl;
} // get2D3D


void XBuilder::sfm()
{
    // 0. gray scale conversion
    
    // 1. keypoint + descriptor
    
    vector<pair<int, std::pair<int,int> > > pairs = KeyPoint_FMatrix_Matching ();
    
    // 2. 2-view reconstruction
    Two_View_Reconstruction(pairs);
    
    
    fileSave("result3D-2View.txt");

    // reconstruction for other views

    while (images_processed.size() != images.size())
        {
        // find the image of highest correspondences among the images not processed,
        vector<Point3f> pts3;
        vector<Point2f> pts2;
        size_t nCorrespMax=0;
        int image_selected=0;
        for (int i=0; i<images.size(); i++)
            {
            if (!(images_processed.find(i) == images_processed.end()))
                continue;
            
            // images[i] is not used
            cerr << "-- image for insertion: " << i << endl;
            
            vector<Point3f> q3;
            vector<Point2f> q2;
            
            get2D3D (q2,q3, i,
                     images_processed, matches_pairs, x3d, imgKeypts);

            cerr << "\t q2/q3 has " << q2.size() << endl;
            
            if (q2.size() > nCorrespMax)
                {
                nCorrespMax = q2.size();
                pts2 = q2; pts3 = q3;
                image_selected = i;
                }
            }
        cerr << "Pose estimation for image " << image_selected << " ("
             << image_names[image_selected] << ") with " << pts2.size() << " 2d-3d" << endl;
        
        if (pts2.size() < 10)
            {
            cerr << " 2d-3d correspondences small" << endl;
            continue;
            }
        
        
        Mat_<double> rvec, tvec;
        vector<uchar> inliers;
        bool r = cv::solvePnPRansac(pts3, pts2, K, distortion_coeff, rvec, tvec,
                           false,
                           1000,
                           f_ransac_threshold,
                           0.5 * (double)(pts2.size()/*minInliersCount*/),
                           inliers,
                           CV_EPNP);
        cv::Mat_<double> Rmat;
        cv::Rodrigues(rvec, Rmat);
        
        cerr << " solvePnPRansac() found " << cv::countNonZero(inliers) << " inliers." << endl;
        cerr << "R:" << Rmat << endl << "t:" << tvec << endl;
        break;
        } // while
}
