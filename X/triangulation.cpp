//
//  triangulation.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 24..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"


static
void makePmat(cv::Mat_<double>& P1, cv::Mat_<double> R1, cv::Mat_<double> t1)
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


static
vector<double> reprojecionError (cv::Mat K,
                                 cv::Mat R, cv::Mat t,
                                 vector<cv::Point3d>& X3,
                                 vector<cv::Point2f>& pt1)
{
    vector<double> rms;
    for (int i=0; i<pt1.size(); i++)
        {
        cv::Mat_<double> x3 (X3[i]);
        Mat_<double> p3 = R * x3 + t;
        p3 /= p3(2);
        p3 = K * p3;
        
        double e1 = p3(0) - pt1[i].x;
        double e2 = p3(1) - pt1[i].y;
        
        rms.push_back( sqrt(e1*e1 + e2*e2) );
        }
    return rms;
}


/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */

cv::Mat_<double>
LinearLSTriangulation
(Point3d u,		//homogenous image point (u,v,1)
 Matx34d P,		//camera 1 matrix
 Point3d u1,		//homogenous image point in 2nd camera
 Matx34d P1		//camera 2 matrix
)
{
    
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    //	cout << "u " << u <<", u1 " << u1 << endl;
    //	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u.x*P(1)-u.y*P(0);
    //	A(3) = u1.x*P1(2)-P1(0);
    //	A(4) = u1.y*P1(2)-P1(1);
    //	A(5) = u1.x*P(1)-u1.y*P1(0);
    //	Matx43d A; //not working for some reason...
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u1.x*P1(2)-P1(0);
    //	A(3) = u1.y*P1(2)-P1(1);
    Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
              u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
              u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
              u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
              );
    Matx41d B(-(u.x*P(2,3)	-P(0,3)),
              -(u.y*P(2,3)	-P(1,3)),
              -(u1.x*P1(2,3)	-P1(0,3)),
              -(u1.y*P1(2,3)	-P1(1,3)));
    
    Mat_<double> X;
    solve(A,B,X,DECOMP_SVD);
    
    return X;
}



/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
#define EPSILON 0.0001

Mat_<double>
IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
                               Matx34d P,			//camera 1 matrix
                               Point3d u1,			//homogenous image point in 2nd camera
                               Matx34d P1			//camera 2 matrix
)
{
    double wi = 1, wi1 = 1;
    Mat_<double> X(4,1);
    for (int i=0; i<10; i++)
        { //Hartley suggests 10 iterations at most
            Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
            X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
            
            //recalculate weights
            double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
            double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
            
            //breaking point
            if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;
            
            wi = p2x;
            wi1 = p2x1;
            
            //reweight equations and solve
            Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,
                      (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,
                      (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,
                      (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
                      );
            Mat_<double> B = (Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
                              -(u.y*P(2,3)	-P(1,3))/wi,
                              -(u1.x*P1(2,3)	-P1(0,3))/wi1,
                              -(u1.y*P1(2,3)	-P1(1,3))/wi1
                              );
            
            solve(A,B,X_,DECOMP_SVD);
            X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
        }
    return X;
}

//Triagulate points
double TriangulatePoints(const vector<Point2f>& pt1,
                         const vector<Point2f>& pt2,
                         const Mat& K,
                         const Mat& Kinv,
                         const Mat& distcoeff,
                         const Matx34d& P,
                         const Matx34d& P1,
                         vector<Point3d>& pointcloud)
{
    cout << "Triangulating...";

    vector<double> reproj_error;
    
    Mat_<double> KP1 = K * Mat(P1);

    for (int i=0; i<pt1.size(); i++)
        {
        Point3d u(pt1[i].x,pt1[i].y,1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u.x = um(0); u.y = um(1); u.z = um(2);
        
        Point2f kp1 = pt2[i];
        Point3d u1(kp1.x,kp1.y,1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);
        
        Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);
        
        
        Mat_<double> xPt_img = KP1 * X;	//reproject
//		cout <<	"Point * K: " << xPt_img << endl;
        Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
        
        double reprj_err = norm(xPt_img_-kp1);
        reproj_error.push_back(reprj_err);
        
//        pointcloud.push_back(X);
        }
    
    Scalar mse = mean(reproj_error);
    cout << "- Done ("<<pointcloud.size()<<"points, "
         << " mean reproj err = " << mse[0] << ")"<< endl;
    
    return mse[0];
}

// --------------------------------------------------------------------------------------------


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
                                const Mat& Kinv, bool flag=false)
{
#if 1
    Mat_<double> R1t = Mat_<double>(Rs[0]).t();
    Mat_<double> R2t = Mat_<double>(Rs[1]).t();
    Mat_<double> m1 = (Mat_<double>(3,1) << ps[0].x, ps[0].y, 1);
    Mat_<double> m2 = (Mat_<double>(3,1) << ps[1].x, ps[1].y, 1);
    Mat_<double> K1 = R1t*(Kinv*m1), K2 = R2t*(Kinv*m2);
    Mat_<double> B1 = -R1t*Mat_<double>(ts[0]);
    Mat_<double> B2 = -R2t*Mat_<double>(ts[1]);

    return findRayIntersection(*K1.ptr<Point3d>(), *B1.ptr<Point3d>(),
                               *K2.ptr<Point3d>(), *B2.ptr<Point3d>());
#else
    
    Point3d u(ps[0].x,ps[0].y,1.0);
    Mat_<double> um = Kinv * Mat_<double>(u);
    u.x = um(0); u.y = um(1); u.z = um(2);
    
    Point2f kp1 = ps[1];
    Point3d u1(kp1.x,kp1.y,1.0);
    Mat_<double> um1 = Kinv * Mat_<double>(u1);
    u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);
    
    cv::Mat_<double> P0(3,4), P1(3,4);
    makePmat(P0, Rs[0], ts[0]);
    makePmat(P1, Rs[1], ts[1]);
    Mat_<double> X = IterativeLinearLSTriangulation(u,P0,u1,P1);

    if (flag) {
        cerr << "X=" << X << endl;
        cerr << "P0=" << P0 << endl;
        cerr << "P1=" << P1 << endl;
        cerr << "u0=" << u << endl;
        cerr << "u1=" << u1 << endl;
    }

    return Point3d(X(0), X(1), X(2));
#endif
}




int XBuilder::triangulate (cv::Mat R0, cv::Mat t0,
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
        
        cv::Point3d p3 = triangulatePoint(ps, Rs, ts, this->Kinv, m==0);
        X3.push_back (p3);

        }
    
    vector<double> err = reprojecionError (this->K, Rs[0], ts[0], X3, pt1);
    std::sort(err.begin(), err.end());
    cv::Scalar mse = cv::mean(err);
    cerr << "! mean re-projection error = " << mse[0] << endl;
    cerr << "! max re-projection error = " << err[err.size()-1] << endl;
    cerr << "! median re-projection error = " << err[err.size()/2] << endl;
    
    err = reprojecionError (this->K, Rs[1], ts[1], X3, pt1);
    std::sort(err.begin(), err.end());
    mse = cv::mean(err);
    cerr << "! mean re-projection error = " << mse[0] << endl;
    cerr << "! max re-projection error = " << err[err.size()-1] << endl;
    cerr << "! median re-projection error = " << err[err.size()/2] << endl;
    
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
    
    return (int)(100*ratio);
}


// EOF //