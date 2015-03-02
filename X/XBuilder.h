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
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace cv;


// -------------------------------------------------------------------------------- //
vector<double> reprojecionError (cv::Mat K,
                                 cv::Mat R, cv::Mat t,
                                 vector<cv::Point3d>& X3,
                                 vector<cv::Point2f>& pt1);

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps);
void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps);
void getAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgKeypts1,
                               const std::vector<cv::KeyPoint>& imgKeypts2,
                               const std::vector<cv::DMatch>& matches,
                               std::vector<cv::Point2f>& pt_set1,
                               std::vector<cv::Point2f>& pt_set2);
void getAlignedKeyPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
                                  const std::vector<cv::KeyPoint>& imgpts2,
                                  const std::vector<cv::DMatch>& matches,
                                  std::vector<cv::KeyPoint>& pt_set1,
                                  std::vector<cv::KeyPoint>& pt_set2);
// -------------------------------------------------------------------------------- //


typedef std::map< std::pair<int,int>,  std::vector<cv::DMatch> > MatchMap;

const double triangulation_err_th = 30.0;

const double f_ransac_threshold = 2.50;
const double f_ransac_confidence = 0.999;


struct P3D {
    cv::Point3d X;
    vector<int> ids; // ids[k] : keypoint index in the k-th view
};

struct XBuilder {
    // images
    vector<Mat> images;
    vector<string> image_names;
    set<int> images_processed;
    
    // matching
    vector< vector<cv::KeyPoint> > imgKeypts;
    MatchMap matches_pairs;
    std::map<std::pair<int,int>, cv::Mat> mapF;
    

    // calibration data
    cv::Mat distortion_coeff, K, Kinv;
    
    // 3D reconstructed.
    std::vector<P3D> x3d;
    std::vector<cv::Mat> R; // rotation
    std::vector<cv::Mat> t; // translation
    
//    std::vector<cv::Point3d> pointcloud;
    std::vector<cv::Vec3b>   pointcloud_RGB;
    
    void open_imgs_dir(string image_dir);
    void fileSave(const std::string name);

    void sfm();
    std::vector<pair<int,std::pair<int,int> > > KeyPoint_FMatrix_Matching ();
    int triangulate (cv::Mat R0, cv::Mat t0,
                     cv::Mat  R1, cv::Mat t1,
                     vector<Point2f>& pt1, vector<Point2f>& pt2,
                     vector<Point3d>& X3,
                     vector<uchar>*pinlier=0);
    
    void Two_View_Reconstruction(vector<pair<int, std::pair<int,int> > >& pairs);
    void doBA(); //BA
    void printReprojectionError();
    
    void ba(); // bundle-adjustment
    void ba_test();
};


#endif /* defined(__sfm__XBuilder__) */
