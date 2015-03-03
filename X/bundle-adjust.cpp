//
//  bundle-adjust.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 27..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

// based on ceres-solver/suite-sparse
// $ brew install ceres-solver

#include <opencv2/opencv.hpp>
#include "XBuilder.h"
#include "ceres-ba.h"
#include "ceres-pose.h"

void XBuilder::ba_test()
{
    std::map<int/*old_index*/, int/*new_index*/> new_index;
    std::map<int, int> old_index;
    int num_observations = 0;
    for (int i=0; i<this->x3d.size(); i++)
        {
        vector<int> &ids = x3d[i].ids;
        int new_index_c = 0;
        for (int m=0; m<ids.size(); m++)
            if (ids[m]>=0)
                {
                new_index[m] = new_index_c ++;
                num_observations ++;
                }
        }
    for (map<int,int>::iterator it=new_index.begin(); it!=new_index.end(); ++it)
        old_index[it->second/*new index*/] = it->first;/*old index*/
    
    cerr << "- index_map has " << new_index.size() << " elem." << endl;
    for (int old_index=0; old_index < images.size(); old_index++)
        {
        map<int,int>::iterator pid = new_index.find(old_index);
        if (pid != new_index.end())
            cerr << "   old index " << old_index << " will be mapped to " << pid->second << endl;
        else
            cerr << "   old index " << old_index << " was not used.\n" ;
        }
    
    // file save to the format of ceres //
    ofstream fs ("problem-2.txt");
    fs << new_index.size() << ' ' << x3d.size() << ' ' << num_observations << endl;
    // observation
    for (int i=0; i<this->x3d.size(); i++)
        {
        vector<int> &ids = x3d[i].ids;
        for (int m=0; m<ids.size(); m++)
            if (ids[m]>=0)
                {
                int new_cam_id = new_index[m];
                int str_id = i;
                fs << new_cam_id << ' ' << str_id << ' '
                << (float)imgKeypts[m][ids[m]].pt.x << ' ' << (float)imgKeypts[m][ids[m]].pt.y << endl;
                }
        }
    fs.precision(20);
    fs << std::scientific;
    // motion
    for (map<int,int>::iterator it=old_index.begin(); it!=old_index.end(); ++it)
        {
        int old_id = it->second;
        cerr << " new index " << it->first << " goes to the old index " << old_id << endl;
        cv::Mat_<double> rvec(3,1);
        cv::Rodrigues(this->R[old_id], rvec);
        cerr << "R:" << R[old_id] <<endl;
        cerr << "->" << rvec << endl;
        cv::Mat_<double> Rm(3,3);
        cv::Rodrigues(rvec, Rm);
        cerr << "->" << Rm << endl;
        
        fs << rvec(0) << endl << rvec(1) << endl << rvec(2) << endl;
        fs << this->t[old_id].at<double>(0) << endl
        << this->t[old_id].at<double>(1) << endl
        << this->t[old_id].at<double>(2) << endl ;
        }
    // structure
    for (int i=0; i<this->x3d.size(); i++)
        fs << x3d[i].X.x << endl << x3d[i].X.y << endl << x3d[i].X.z << endl;
    // cam intrinsic
    fs << std::fixed;
    fs << this->K.at<double>(1,1) << endl
    << this->K.at<double>(0,2) << endl
    << this->K.at<double>(1,2) << endl;
    // end.
    fs.close();
    
    // read-file
    //
    {
    string filename = "problem-2.txt.ba.txt";
    ifstream fs (filename);
    if (fs.is_open()==false)
        {
        cerr << "file open failed: " << filename << endl;
        exit (0);
        }
    int num_cam, num_str, num_observ;
    fs >> num_cam >>  num_str >> num_observ;
    // observation
    for (int i=0; i<this->x3d.size(); i++)
        {
        vector<int> &ids = x3d[i].ids;
        for (int m=0; m<ids.size(); m++)
            if (ids[m]>=0)
                {
                int new_cam_id;
                int str_id = i;
                float x, y;
                fs >> new_cam_id >> str_id >> x >> y;
                }
        }
    // motion
    for (map<int,int>::iterator it=old_index.begin(); it!=old_index.end(); ++it)
        {
        int old_id = it->second;
        cerr << " new index " << it->first << " goes to the old index " << old_id << endl;
        
        cv::Mat_<double> rvec(3,1);
        fs >> rvec(0) >> rvec(1) >> rvec(2);
        cv::Rodrigues(rvec, this->R[old_id]);
        
        fs >> this->t[old_id].at<double>(0)
        >> this->t[old_id].at<double>(1)
        >> this->t[old_id].at<double>(2);
        }
    // structure
    for (int i=0; i<this->x3d.size(); i++)
        fs >> x3d[i].X.x
        >> x3d[i].X.y
        >> x3d[i].X.z ;
    
    // cam intrinsic
    fs >> this->K.at<double>(1,1)
    >> this->K.at<double>(0,2)
    >> this->K.at<double>(1,2);
    this->K.at<double>(0,0) = this->K.at<double>(1,1);
    // end.
    fs.close();
    }
    return;
}


void XBuilder::ba_pose(cv::Mat_<double> r, cv::Mat_<double> t, vector<Point2f>& p2, vector<Point3f>& p3)
{
    cerr << "*** ba_pose() " << endl;
    cerr << "r=" << r.t() << " t=" << t.t() << endl;
    
    vector<double> rv, tv;
    for (int i=0; i<3; i++)
        rv.push_back(r(i)), tv.push_back(t(i));
    
    vector<ObsPose> obs;
    for (int i=0; i<p2.size(); i++)
        {
        ObsPose o;
        o.p.push_back(p2[i].x);
        o.p.push_back(p2[i].y);
        o.X.push_back(p3[i].x);
        o.X.push_back(p3[i].y);
        o.X.push_back(p3[i].z);
        obs.push_back(o);
        }

    CeresPose (obs, rv, tv,
               this->K.at<double>(0,0), this->K.at<double>(0,2), this->K.at<double>(1,2));
    
    
    for (int i=0; i<3; i++)
        r(i) = rv[i], t(i) = tv[i];

    cerr << "*** ba_pose finished:" << endl
    << "r=" << r.t() << " t=" << t.t() << endl;
}

void XBuilder::ba()
{
    std::map<int/*old_index*/, int/*new_index*/> new_index;
    std::map<int, int> old_index;
    int num_observations = 0;
    for (int i=0; i<this->x3d.size(); i++)
        {
        vector<int> &ids = x3d[i].ids;
        int new_index_c = 0;
        for (int m=0; m<ids.size(); m++)
            if (ids[m]>=0)
                {
                new_index[m] = new_index_c ++;
                num_observations ++;
                }
        }
    for (map<int,int>::iterator it=new_index.begin(); it!=new_index.end(); ++it)
        old_index[it->second/*new index*/] = it->first;/*old index*/
    
    cerr << "- index_map has " << new_index.size() << " elem." << endl;
    for (int old_index=0; old_index < images.size(); old_index++)
        {
        map<int,int>::iterator pid = new_index.find(old_index);
        if (pid != new_index.end())
            cerr << "   old index " << old_index << " will be mapped to " << pid->second << endl;
        else
            cerr << "   old index " << old_index << " was not used.\n" ;
        }
    
    double nc, np, no;
    vector<double> motion_str;
    vector<Observation> obs;
    double focal, x0, y0;
    
    nc = new_index.size();
    np = x3d.size() ;
    no = num_observations;
    
    // observation
    for (int i=0; i<this->x3d.size(); i++)
        {
        vector<int> &ids = x3d[i].ids;
        for (int m=0; m<ids.size(); m++)
            if (ids[m]>=0)
                {
                int new_cam_id = new_index[m];
                int str_id = i;
                Observation o;
                o.cam_id = new_cam_id;
                o.str_id = str_id;
                o.pt.push_back(imgKeypts[m][ids[m]].pt.x);
                o.pt.push_back(imgKeypts[m][ids[m]].pt.y);
                obs.push_back(o);
                }
        }
    // motion + structure
    for (map<int,int>::iterator it=old_index.begin(); it!=old_index.end(); ++it)
        {
        int old_id = it->second;
        cv::Mat_<double> rvec(3,1);
        cv::Rodrigues(this->R[old_id], rvec);
        
        motion_str.push_back(rvec(0));
        motion_str.push_back(rvec(1));
        motion_str.push_back(rvec(2));
        
        motion_str.push_back(this->t[old_id].at<double>(0));
        motion_str.push_back(this->t[old_id].at<double>(1));
        motion_str.push_back(this->t[old_id].at<double>(2));
        }
    // structure
    for (int i=0; i<this->x3d.size(); i++)
        {
        motion_str.push_back(x3d[i].X.x);
        motion_str.push_back(x3d[i].X.y);
        motion_str.push_back(x3d[i].X.z);
        }
    // cam intrinsic
    focal = this->K.at<double>(1,1);
    x0 = this->K.at<double>(0,2);
    y0 = this->K.at<double>(1,2);
    
    //
    //  run bundle-adjustment
    //
    
    Bundler(nc, np, no, obs, motion_str, focal, x0, y0);
    
    
    
    // recover the updated motion + str + cam_intrinsic
    
    // motion + structure
    int n=0;
    for (map<int,int>::iterator it=old_index.begin(); it!=old_index.end(); ++it)
        {
        cv::Mat_<double> rvec(3,1);
        for (int k=0; k<3; k++)
            rvec(k) = motion_str[n++];
        
        int old_id = it->second;
        cv::Rodrigues(rvec, this->R[old_id]);
        
        for (int k=0; k<3; k++)
            this->t[old_id].at<double>(k) = motion_str[n++];
        }
    // structure
    for (int i=0; i<this->x3d.size(); i++)
        {
        x3d[i].X.x = motion_str[n++];
        x3d[i].X.y = motion_str[n++];
        x3d[i].X.z = motion_str[n++];
        }
    // cam intrinsic
    this->K.at<double>(0,0) = this->K.at<double>(1,1) = focal;
    this->K.at<double>(0,2) = x0;
    this->K.at<double>(1,2) = y0;
    
    return;
}


// EOF //