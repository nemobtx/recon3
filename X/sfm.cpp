//
//  sfm.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"

bool sort_descending(pair<int,pair<int,int> > a, pair<int,pair<int,int> > b) { return a.first > b.first; }


void XBuilder::printReprojectionError()
{
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
}


//
// OpenCV contrib module is used for sparse bundle adjustment //
//

int totalMeasurements (vector<P3D>& p3)
{
    int n=0;
    for (int i=0; i<p3.size(); i++)
        for (int j=0; j<p3[i].ids.size(); j++)
            if (p3[i].ids[j] >= 0) // valid measurement index exists
                n++;
    return n;
}

void XBuilder::doBA()
{
    cerr << "! Bundle Adjustment by OpenCV: Contrib Module" << endl;

    const int M = (int)this->x3d.size();
    const int N = (int)images.size();
    const int Nmeas = totalMeasurements (this->x3d);
    
    // positions of points in global coordinate system (input and output)
    vector<Point3d> points(M);
    // projections of 3d points for every camera
    vector< vector<Point2d> > imagePoints(N,vector<Point2d>(M));
    // visibility of 3d points for every camera
    vector< vector<int> >   visibility(N, vector<int>(M));
    // intrinsic matrices of all cameras (input and output)
    vector<Mat> cameraMatrix(N);
    // rotation matrices of all cameras (input and output)
    vector<Mat> R(N);
    // translation vector of all cameras (input and output)
    vector<Mat> T(N);
    // distortion coefficients of all cameras (input and output)
    vector<Mat> distCoeffs(0);
    
    int num_global_cams = (int)this->images.size(); // pointcloud[0].imgpt_for_img.size();
    vector<int> global_cam_id_to_local_id(num_global_cams,-1);
    vector<int> local_cam_id_to_global_id(N,-1);
    int local_cam_count = 0;
    
    for (int pt3d = 0; pt3d < this->x3d.size(); pt3d++)
        {
        points[pt3d] = this->x3d[pt3d].X;
        
        if (pt3d<10)
            {
            cerr << "-" << endl;
            cerr << "X:" << points[pt3d] << endl;
            }
        
        for (int pt3d_img = 0; pt3d_img < num_global_cams; pt3d_img++)
            {
            int indx = this->x3d[pt3d].ids[pt3d_img];
            if (indx >= 0) // a measurement is from view[indx]
                {
                if (global_cam_id_to_local_id[pt3d_img] < 0) // if not initialized
                    {
                    local_cam_id_to_global_id[local_cam_count] = pt3d_img;
                    global_cam_id_to_local_id[pt3d_img] = local_cam_count++;
                    }
                
                int local_cam_id = global_cam_id_to_local_id[pt3d_img];
                
                //2d point
                Point2d pt2d_for_pt3d_in_img = this->imgKeypts[pt3d_img][indx].pt;
                imagePoints[local_cam_id][pt3d] = pt2d_for_pt3d_in_img;

                if (pt3d<10)
                    cerr << "p2:" << pt2d_for_pt3d_in_img << endl;
                
                //visibility in this camera
                visibility[local_cam_id][pt3d] = 1;
                }
            }
        
        //2nd pass to mark not-founds
        for (int pt3d_img = 0; pt3d_img < num_global_cams; pt3d_img++)
            {
            int indx = this->x3d[pt3d].ids[pt3d_img];
            if (indx < 0)
                {
                //see if this global camera is being used locally in the BA
                vector<int>::iterator local_it = std::find(local_cam_id_to_global_id.begin(),local_cam_id_to_global_id.end(),pt3d_img);
                if (local_it != local_cam_id_to_global_id.end())
                    {
                    //this camera is used, and its local id is:
                    int local_id = (int)(local_it - local_cam_id_to_global_id.begin());
                    
                    if (local_id >= 0)
                        {
                        //reproject
                        Mat_<double> X (this->x3d[pt3d].X);
                        
                        Mat_<double> xPt_img = this->K * (this->R[pt3d]*X + this->t[pt3d]);
                        Point2d xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
                        
                        imagePoints[local_id][pt3d] = xPt_img_; //TODO reproject point on this camera
                        visibility[local_id][pt3d] = 0;
                    }
                }
            }
        }
    }
    
    // notice
    // the following is just a soft copy.
    // the update of BA is automatically 반영된다.
    for (int i=0; i<N; i++)
        {
        cameraMatrix[i] = this->K;
        int indx = local_cam_id_to_global_id[i];
        if (indx<0) // this camera will not be used. so, set arbitrarily.
            {
            R[i] = (cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,1);
            t[i] = (cv::Mat_<double>(3,1) << 0,0,0);
            }
        else
            {
            R[i] = this->R[indx];
            T[i] = this->t[indx];
            }
        //              distCoeffs[i] = Mat(); //::zeros(4,1, CV_64FC1);
        }
    
    cerr << "R[0]:" << this->R[0] << endl;
    
    cout << "Adjust bundle... \n";
    cv::LevMarqSparse::bundleAdjust(points, imagePoints, visibility,
                                    cameraMatrix, R, T, distCoeffs);
    cout << "DONE\n";
    
    // get the BAed 3D points
    for (int pt3d = 0; pt3d < this->x3d.size(); pt3d++)
        {
        this->x3d[pt3d].X = points[pt3d];
        }
    
    // get the BAed cameras
    // well, we don't have to, thanks to the soft-copy
    cerr << "R[0]BA:" << this->R[0] << endl;
}




void XBuilder::sfm()
{
    // 0. gray scale conversion
    
    // 1. keypoint + descriptor
    
    std::pair<int,int> min_pair = KeyPoint_FMatrix_Matching ();
    
    cerr << endl;
    cerr << "! min_pair=" << min_pair.first << ", " << min_pair.second << endl;
    cerr << endl;
    
    //
    // 2-view reconstruction
    //
    vector<cv::Point2f> pt1, pt2;
    getAlignedPointsFromMatch(imgKeypts[min_pair.first], imgKeypts[min_pair.second], matches_pairs[min_pair], pt1, pt2);
    
    
    // re-do RANSAC for the two views
    vector<uchar> isInlier(pt1.size());
    cv::Mat F2 = cv::findFundamentalMat(pt1, pt2,
                                       FM_RANSAC,
                                       f_ransac_threshold/*pixel threshold*/,
                                       0.99,
                                       isInlier);
    if (cv::countNonZero(isInlier) < isInlier.size())
        {
        cerr << " new F-RANSAC finds " << cv::countNonZero(isInlier) << " out of " << isInlier.size() << endl;
        vector<DMatch> matches2;
        for (int i=0; i<isInlier.size(); i++)
            if (isInlier[i])
                {
                matches2.push_back(matches_pairs[min_pair][i]);
                }
        cerr << " before: " << matches_pairs[min_pair].size() ;
        matches_pairs[min_pair] = matches2;
        cerr << " after: " << matches_pairs[min_pair].size() ;
        cerr << endl;
        pt1.clear(); pt2.clear();
        getAlignedPointsFromMatch(imgKeypts[min_pair.first], imgKeypts[min_pair.second], matches_pairs[min_pair], pt1, pt2);
        this->mapF[min_pair] = F2;
        }
    //
    cv::Mat_<double> F = this->mapF[min_pair];
    
    cv::Mat_<double> E = K.t() * F * K;
    if (fabs( cv::determinant(E) ) > 1E-07)
        {
        fprintf(stderr, "det(E) = %.3le\n", fabs(cv::determinant(E)));
        }
    fprintf(stderr, "! det(E) = %.3le\n", fabs(cv::determinant(E)));
    cv::SVD svd(E);
    cerr << " svd.u= " << endl << svd.u << endl;
    cerr << " svd.w= " << endl << svd.w << endl;
    cerr << " svd.vt= " << endl << svd.vt << endl;

    // check the ratio of the two largest singular values
    double s_ratio = svd.w.at<double>(0) / svd.w.at<double>(1);
    if (s_ratio < 0.7)
        {
        cerr << "! s-ratio is too small\n";
        }
    cv::Mat_<double> W = (cv::Mat_<double>(3,3) <<	0.,-1,0, 1,0,0, 0,0,1);
    cerr << "W = " << W << endl;
    
    cv::Mat_<double> R1, R2, t1, t2;
    R1 = svd.u * W * svd.vt;
    R2 = svd.u * W.t() * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
    
    if (cv::determinant(R1) < -0.7) // if it's det is -1
        {
        cerr << " det(R1) == " << cv::determinant(R1) << endl;
        R1 *= -1;
        }
    if (cv::determinant(R2) < -0.7) // if it's det is -1
        {
        cerr << " det(R2) == " << cv::determinant(R2) << endl;
        R2 *= -1;
        }
    
    cv::Mat_<double> R0 = (cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,1);
    cv::Mat_<double> t0 = (cv::Mat_<double>(3,1) << 0,0,0);
    
    // allocation
    this->x3d.resize(pt1.size());
    
    vector<cv::Point3d> X3; // 3D reconstructed
    
    vector<cv::Mat> Rs; Rs.push_back(R1); Rs.push_back(R2);
    vector<cv::Mat> ts; ts.push_back(t1); ts.push_back(t2);
    
    std::vector< std::pair<int, pair<int,int> > > pratio;
    for (int i=0; i < Rs.size(); i++)
        {
        for (int j=0; j < ts.size(); j++)
            {
            cerr << "positive test of " << i << " and " << j << endl;
            int pos = triangulate (R0, t0, Rs[i], ts[j], pt1, pt2, X3);
            pratio.push_back(make_pair(pos, make_pair(i,j)));
            cerr << "-------" << endl;
            }
        }
    
    std::sort(pratio.begin(), pratio.end(), sort_descending);
    for (int i=0; i<pratio.size(); i++)
        cerr << " pratio-" << i << " = " << pratio[i].first << endl;
    
    if (pratio[0].first < 50)
        {
        cerr << "!! No positve reconstruction was obtained. Check!" << endl;
        exit (1);
        }
    
    int r_id = pratio[0].second.first;
    int t_id = pratio[0].second.second;

    cerr << "R* =" << Rs[r_id] << endl << "t* =" << ts[t_id] << endl;
    
    cerr << "--------- triangulation ----------" << endl;
    int pos = triangulate (R0, t0, Rs[r_id], ts[t_id], pt1, pt2, X3);
    cerr << "Positive Ratio = " << pos << endl;
    
    // record
    for (int m=0; m<pt1.size(); m++)
        {
        this->x3d[m].X = X3[m];
        this->x3d[m].ids.resize(this->image_names.size(), -1);
        this->x3d[m].ids[min_pair.first] =matches_pairs[min_pair][m].queryIdx;
        this->x3d[m].ids[min_pair.second]=matches_pairs[min_pair][m].trainIdx;
        }
    
    this->R.resize(images.size());
    this->t.resize(images.size());
    this->R[min_pair.first] = R0;
    this->R[min_pair.second] = Rs[r_id];
    this->t[min_pair.first] = t0;
    this->t[min_pair.second] = ts[t_id];

    this->images_processed.resize(images.size(), false);
    this->images_processed[min_pair.first] = true;
    this->images_processed[min_pair.second] = true;
    
    for (int i=0; i<10; i++)
        cerr << "X:" << X3[i] << endl;
    for (int i=0; i<this->R.size(); i++)
        cerr << "R:"<<this->R[i] << endl << "t:" << this->t[i] << endl;

    // the rest of the cams are packed with default; just for BA.
    for (int i=0; i<images.size(); i++)
        if (!(i==min_pair.first && i==min_pair.second))
            {
            this->R[i] = R0.clone();
            this->t[i] = t0.clone();
            }

    printReprojectionError();

    // now initial 2d reconstruction is ready.
    // do BA
    doBA ();
    
    // check the result of BA
    printReprojectionError();
    
    
    
}
