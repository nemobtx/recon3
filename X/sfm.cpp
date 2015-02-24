//
//  sfm.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"

static
void makePmat(cv::Mat_<double>& P1, cv::Mat_<double>& R1, cv::Mat_<double>& t1)
{
    for (int r=0; r<3; r++)
        {
        int c=0;
        for (; c<3; c++)
            P1(r,c) = R1(r,c);
        P1(r,c) = t1(r);
        }
    return;
}

static Point3d findRayIntersection(Point3d k1, Point3d b1, Point3d k2, Point3d b2)
{
    double a[4], b[2], x[2];
    a[0] = k1.dot(k1);
    a[1] = a[2] = -k1.dot(k2);
    a[3] = k2.dot(k2);
    b[0] = k1.dot(b2 - b1);
    b[1] = k2.dot(b1 - b2);
    Mat_<double> A(2, 2, a), B(2, 1, b), X(2, 1, x);
    solve(A, B, X);
    
    double s1 = X.at<double>(0, 0);
    double s2 = X.at<double>(1, 0);
    return (k1*s1 + b1 + k2*s2 + b2)*0.5f;
}

static Point3d triangulatePoint(const vector<Point2d>& ps,
                                const vector<Mat>& Rs,
                                const vector<Mat>& ts,
                                const Mat& Kinv)
{
    Mat_<double> R1t = Mat_<double>(Rs[0]).t();
    Mat_<double> R2t = Mat_<double>(Rs[1]).t();
    Mat_<double> m1 = (Mat_<double>(3,1) << ps[0].x, ps[0].y, 1);
    Mat_<double> m2 = (Mat_<double>(3,1) << ps[1].x, ps[1].y, 1);
    Mat_<double> K1 = R1t*(Kinv*m1), K2 = R2t*(Kinv*m2);
    Mat_<double> B1 = -R1t*Mat_<double>(ts[0]);
    Mat_<double> B2 = -R2t*Mat_<double>(ts[1]);
    return findRayIntersection(*K1.ptr<Point3d>(), *B1.ptr<Point3d>(),
                               *K2.ptr<Point3d>(), *B2.ptr<Point3d>());
}

vector<double> reprojecionError (cv::Mat& K,
                         vector< cv::Mat >& Rs, vector< cv::Mat >& ts,
                         vector<cv::Point3d>& X3,
                         vector<cv::Point2f>& pt1, vector<cv::Point2f>& pt2)
{
    vector<double> rms;
    for (int i=0; i<pt1.size(); i++)
        {
        cv::Mat_<double> x3 (X3[i]);
        Mat_<double> p3 = Rs[0] * x3 + ts[0];
        Mat_<double> q3 = Rs[1] * x3 + ts[1];
        p3 /= p3(2);
        q3 /= q3(2);
        p3 = K * p3;
        q3 = K * q3;
        
        double e1 = p3(0) - pt1[i].x;
        double e2 = p3(1) - pt1[i].y;
        double e3 = q3(0) - pt2[i].x;
        double e4 = q3(1) - pt2[i].y;
        
        rms.push_back ( (sqrt(e1*e1 + e2*e2) + sqrt(e3*e3 + e4*e4))/2. );
        }
    return rms;
}


bool XBuilder::triangulate (cv::Mat R0, cv::Mat t0,
                            cv::Mat  R1, cv::Mat t1,
                            vector<Point2f>& pt1, vector<Point2f>& pt2,
                            vector<Point3d>& X3)
{
    X3.clear(); // returned, result of triangulation
    
    vector< cv::Mat > Rs, ts;
    Rs.push_back(R0);
    Rs.push_back(R1);
    ts.push_back(t0);
    ts.push_back(t1);
    for (int m=0; m<pt1.size(); m++)
        {
        cv::Point2d p(pt1[m].x, pt1[m].y);
        cv::Point2d q(pt2[m].x, pt2[m].y);
        vector<Point2d> ps;
        ps.push_back(p);
        ps.push_back(q);
        
        cv::Point3d p3 = triangulatePoint(ps, Rs, ts, this->Kinv);
        X3.push_back (p3);
        }
    
    vector<double> err = reprojecionError (this->K, Rs, ts, X3, pt1, pt2);
    std::sort(err.begin(), err.end());
//    for (int m=0; m<err.size(); m++)
//        {
//        cerr << err[m] << " : ";
//        for (int i=0; i<this->images.size(); i++)
//            cerr << this->x3d[m].ids[i] << "  ";
//        cerr << endl;
//        }
    cv::Scalar mse = cv::mean(err);
    cerr << "! mean re-projection error = " << mse[0] << endl;

    // check the polarity of the reconstruction: front or back of the camera
    vector<int> flag(pt1.size(), 0);
    for (int i=0; i<pt1.size(); i++)
        {
        cv::Mat_<double> x3(X3[i]);
        cv::Mat_<double> p3 = R1*x3 + t1;
        cv::Mat_<double> q3 = R0*x3 + t0;
        if (p3(2)>0 && q3(2)>0)
            flag[i] = 1; // good, in front of the two cameras
        }
    
    double ratio = cv::countNonZero(flag) / (double)X3.size();
    
    cerr << "! positive ratio = " << ratio << endl;
    
    if (ratio < 0.7) return false;
    return true;
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

    const int M = this->x3d.size();
    const int N = images.size();
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
    
    int num_global_cams = pointcloud[0].imgpt_for_img.size();
    vector<int> global_cam_id_to_local_id(num_global_cams,-1);
    vector<int> local_cam_id_to_global_id(N,-1);
    int local_cam_count = 0;
    
    for (int pt3d = 0; pt3d < pointcloud.size(); pt3d++) {
        points[pt3d] = pointcloud[pt3d].pt;
        //              imagePoints[pt3d].resize(N);
        //              visibility[pt3d].resize(N);
        
        for (int pt3d_img = 0; pt3d_img < num_global_cams; pt3d_img++) {
            if (pointcloud[pt3d].imgpt_for_img[pt3d_img] >= 0) {
                if (global_cam_id_to_local_id[pt3d_img] < 0)
                    {
                    local_cam_id_to_global_id[local_cam_count] = pt3d_img;
                    global_cam_id_to_local_id[pt3d_img] = local_cam_count++;
                    }
                
                int local_cam_id = global_cam_id_to_local_id[pt3d_img];
                
                //2d point
                Point2d pt2d_for_pt3d_in_img = imgpts[pt3d_img][pointcloud[pt3d].imgpt_for_img[pt3d_img]].pt;
                imagePoints[local_cam_id][pt3d] = pt2d_for_pt3d_in_img;
                
                //visibility in this camera
                visibility[local_cam_id][pt3d] = 1;
            }
        }
        //2nd pass to mark not-founds
        for (int pt3d_img = 0; pt3d_img < num_global_cams; pt3d_img++) {
            if (pointcloud[pt3d].imgpt_for_img[pt3d_img] < 0) {
                //see if this global camera is being used locally in the BA
                vector<int>::iterator local_it = std::find(local_cam_id_to_global_id.begin(),local_cam_id_to_global_id.end(),pt3d_img);
                if (local_it != local_cam_id_to_global_id.end()) {
                    //this camera is used, and its local id is:
                    int local_id = local_it - local_cam_id_to_global_id.begin();
                    
                    if (local_id >= 0) {
                        //reproject
                        Mat_<double> X = (Mat_<double>(4,1) << pointcloud[pt3d].pt.x, pointcloud[pt3d].pt.y, pointcloud[pt3d].pt.z, 1);
                        Mat_<double> P(3,4,Pmats[pt3d_img].val);
                        Mat_<double> KP = cam_matrix * P;
                        Mat_<double> xPt_img = KP * X;
                        Point2d xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
                        
                        imagePoints[local_id][pt3d] = xPt_img_; //TODO reproject point on this camera
                        visibility[local_id][pt3d] = 0;
                    }
                }
            }
        }
    }
    for (int i=0; i<N; i++) {
        cameraMatrix[i] = cam_matrix;
        
        Matx34d& P = Pmats[local_cam_id_to_global_id[i]];
        
        Mat_<double> camR(3,3),camT(3,1);
        camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
        camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
        camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
        R[i] = camR;
        T[i] = camT;
        
        //              distCoeffs[i] = Mat(); //::zeros(4,1, CV_64FC1);
    }
    
    cout << "Adjust bundle... \n";
    cv::LevMarqSparse::bundleAdjust(points,imagePoints,visibility,cameraMatrix,R,T,distCoeffs);
    cout << "DONE\n";
    
    //get the BAed points
    for (int pt3d = 0; pt3d < pointcloud.size(); pt3d++) {
        pointcloud[pt3d].pt = points[pt3d];
    }
    
    //get the BAed cameras
    for (int i = 0; i < N; ++i)
        {
        Matx34d P;
        P(0,0) = R[i].at<double>(0,0); P(0,1) = R[i].at<double>(0,1); P(0,2) = R[i].at<double>(0,2); P(0,3) = T[i].at<double>(0);
        P(1,0) = R[i].at<double>(1,0); P(1,1) = R[i].at<double>(1,1); P(1,2) = R[i].at<double>(1,2); P(1,3) = T[i].at<double>(1);
        P(2,0) = R[i].at<double>(2,0); P(2,1) = R[i].at<double>(2,1); P(2,2) = R[i].at<double>(2,2); P(2,3) = T[i].at<double>(2);
        
        Pmats[local_cam_id_to_global_id[i]] = P;
        }

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
    cv::Mat_<double> F = this->mapF[min_pair];
    vector<cv::Point2f> pt1, pt2;
    getAlignedPointsFromMatch(imgKeypts[min_pair.first], imgKeypts[min_pair.second], matches_pairs[min_pair], pt1, pt2);
    
    cv::Mat_<double> E = K.t() * F * K;
    if (fabs( cv::determinant(E) ) > 1E-07)
        {
        fprintf(stderr, "det(E) = %.3le\n", fabs(cv::determinant(E)));
        }
    fprintf(stderr, "! det(E) = %.3le\n", fabs(cv::determinant(E)));
    cv::SVD svd(E);
    cerr << " svd.w= " << svd.w << endl;
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
    cerr << "R1=" << R1 << endl << "t1=" << t1 << endl;
    
    cv::Mat_<double> P0 = (cv::Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
    cv::Mat_<double> P1(3,4);
    makePmat(P1, R1, t1);
    cerr << " P1=" << P1 << endl;
    
    { // not used
    vector<cv::Point2f> upt1, upt2; // undistorted :=  Kinv * undistort(pt1)
    cv::undistortPoints(pt1, upt1, K, distortion_coeff);
    cv::undistortPoints(pt2, upt2, K, distortion_coeff);
    // cerr << "pt1=" << pt1 << endl << "upt1=" << upt1 << endl;
    }
    // now, triangulation for 3D structure
    
    cv::Mat_<double> R0 = (cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,1);
    cv::Mat_<double> t0 = (cv::Mat_<double>(3,1) << 0,0,0);
    
    // allocation
    this->x3d.resize(pt1.size());
    
    vector<cv::Point3d> X3; // 3D reconstructed
    
    vector<cv::Mat> Rs; Rs.push_back(R1); Rs.push_back(R2);
    vector<cv::Mat> ts; ts.push_back(t1); ts.push_back(t2);
    bool flag = false;
    int i=0, j=0;
    while (i++ < Rs.size() && !flag)
        {
        while (j++ < ts.size() && !flag)
            {
            cerr << "positive test of " << i << " and " << j << endl;
            flag = triangulate (R0, t0, Rs[i], ts[j], pt1, pt2, X3);
            }
        }
    
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
    this->R[min_pair.second] = Rs[i];
    this->t[min_pair.first] = t0;
    this->t[min_pair.second] = ts[j];
    
    // now initial 2d reconstruction is ready.
    // do BA
    doBA ();
}
