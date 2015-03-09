//
//  pnp.cpp
//  sfm
//
//  Created by Yongduek Seo on 3/9/15.
//  Copyright (c) 2015 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"

vector<pair<int,int> >
get2D3D (vector<Point2f>& q2, vector<Point3f>& q3, int new_image_id,
         set<int>& images_processed,
         MatchMap& matches_pairs,
         vector<P3D>& x3d,
         vector<vector<KeyPoint> >& imgKeypts
         )
{
    q2.clear();
    q3.clear();
    vector<bool> usedX3(x3d.size(), false); // indicates whether
    
    vector<pair<int,int> > record;
    
    cerr << "get2d3d()" << endl;
    // find 2d-3D pairs from every possible image pairs
    
    for (set<int>::iterator it=images_processed.begin(); it!=images_processed.end(); it++)
        {
        int used_img_id = *it;
        std::pair<int,int> pair(used_img_id, new_image_id);
        vector<DMatch> matches = matches_pairs[pair];
        int count=0;
        for (int m=0; m<matches.size(); m++)
            {
            for (int i3=0; i3<x3d.size(); i3++)
                if (x3d[i3].ids[used_img_id] == matches[m].queryIdx
                    && usedX3[i3]==false)
                    {
                    q3.push_back(x3d[i3].X);
                    q2.push_back(imgKeypts[new_image_id][matches[m].trainIdx].pt);
                    usedX3[i3]=true;
                    record.push_back (make_pair(i3, matches[m].trainIdx)); // x3d.ids[i3][new_image_id] = matches[m].trainIdx;
                    count++;
                    }
            }
        
        cerr << "\t\t>>  pair(" << used_img_id << "," << new_image_id << ") has " << count << " 2d3d matches" << endl;
        }
    cerr << " --- " << endl;
    return record;
} // get2D3D

cv::Scalar getRandomColor()
{
    static RNG rng(time(0));
    
    return cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

void XBuilder::doPnP()
{
    std::set<int> image4insertion;
    
    while (images_processed.size() != images.size())
        {
        // find the image of highest correspondences among the images not processed,
        
        cv::Mat_<double> Rmat, tvec;
        int image_selected=-1;
        // 1. pose estimation
            {
            vector<Point3f> pts3;
            vector<Point2f> pts2;
            vector<pair<int,int> > record_3d2d;
            size_t nCorrespMax=0;
            for (int i=0; i<images.size(); i++)
                {
                if (!(images_processed.find(i) == images_processed.end()))
                    continue;
    
                // images[i] is not used
                cerr << "-- image for insertion: " << i << endl;

                if (image4insertion.find(i) != image4insertion.end())
                    {
                    cerr << "----------------------------------------" << endl;
                    cerr << "  PnP: Already considered and no good." << endl;
                    cerr << "----------------------------------------" << endl;
                    return;
                    }

                image4insertion.insert(i);
                
                vector<Point3f> q3;
                vector<Point2f> q2;
                
                vector<pair<int,int> > record
                = get2D3D (q2,q3, i/*new_image_id*/,
                           images_processed, matches_pairs, x3d, imgKeypts);
                
                cerr << ">> q2/q3 has " << q2.size() << endl;
                
                if (q2.size() > nCorrespMax)
                    {
                    nCorrespMax = q2.size();
                    pts2 = q2; pts3 = q3;
                    record_3d2d = record;
                    image_selected = i;
                    }
                }
            
            cerr << "Pose estimation for image " << image_selected << " ("
            << image_names[image_selected] << ") with " << pts2.size() << " 2d-3d" << endl;
            
            if (pts2.size() < 100)
                {
                cerr << " 2d-3d correspondences small" << endl;
                continue;
                }
            
            
            Mat_<double> rvec;
            vector<int> inliers;
            Mat mat_inliers;
            double reproj_threshold = 1.2;
            cv::solvePnPRansac(pts3, pts2, K, distortion_coeff, rvec, tvec,
                               false,
                               1000,
                               reproj_threshold,
                               0.95 * (double)pts2.size(),  /*minInliersCount before stop*/
                               inliers, // Output vector that contains indices of inliers
                               CV_EPNP);
            cv::Rodrigues(rvec, Rmat);
            if (fabs((cv::determinant(Rmat)-1.0)) > 1E-7)
                {
                cerr << "check determinamt of Rmat: " << cv::determinant(Rmat) << endl;
                }
            //cerr << " solvePnPRansac() found inliers of size " << inliers.size() << endl;
            
//            if (0) {
//                vector<Point2f> pose2;
//                vector<Point3f> pose3;
//                for (int i=0; i<inliers.size(); i++)
//                    pose2.push_back (pts2[inliers[i]]),
//                    pose3.push_back (pts3[inliers[i]]);
//                
//                vector<double> rep = reprojecionError (this->K,
//                                                       Rmat, tvec,
//                                                       pose3, pose2);
//                sort (rep.begin(), rep.end());
//                //            for (int i=0; i<rep.size(); i+=10)
//                //                cerr << (float)i/(float)rep.size() << "  " << rep[i] << endl;
//                cerr << " -- rms max for inliers = " << rep[rep.size()-1] << endl;
//                cerr << " -- rms min for inliers = " << rep[0] << endl;
//                cerr << " -- rms 50% for inliers (from solvePnPRansac) = " << rep[rep.size()/2] << endl;
//                for (int i=0; i<rep.size(); i++)
//                    cerr << i/(double)rep.size() << "    " << rep[i] << endl;
//            }
            cerr << "R:" << Rmat << endl << "t:" << tvec << endl;
            cerr << " ++++++++++++++++++++ " << endl;
            bool flag_do_bundle = true;
            if (flag_do_bundle)
                {
                vector<Point2f> proj;
                cv::projectPoints(pts3, rvec, tvec, this->K, this->distortion_coeff, proj);
                vector<uchar> outlier(proj.size(), true);
                for (int i=0; i<proj.size(); i++)
                    {
                    double e = sqrt( pow(proj[i].x-pts2[i].x, 2.) + pow(proj[i].y-pts2[i].y,2.) );
                    if (e <= reproj_threshold)
                        outlier[i] = false;
                    }
                cerr << " test shows inliers of " << proj.size() - cv::countNonZero(outlier) << endl;
                
                // register the observations from the inliers, found through solvePnPRansac
                //
                int nRegistered=0;
                for (int i=0; i<record_3d2d.size(); i++)
                    if (outlier[i]==false)
                        {
                        this->x3d[record_3d2d[i].first].ids[image_selected] = record_3d2d[i].second;
                        nRegistered++;
                        }
                cerr << "x3d has been registerd " << nRegistered << " new observations from image " << image_names[image_selected] << endl;
                
                // update the computed motion
                //
                this->R[image_selected] = Rmat.clone();
                this->t[image_selected] = tvec.clone();
                
                cerr << "-------------------------------------------" << endl;
                cerr << "  Before BA " << endl;
                printReprojectionError();
                
                this->ba();
                
                cerr << "-------------------------------------------" << endl;
                cerr << "  ** After BA " << endl;
                printReprojectionError();
                
                for (int i=0; i<images.size(); i++)
                    //if (images_processed.find(i)!=images_processed.end())
                    {
                    cerr << "R" << i << ":" << endl << this->R[i] << endl;
                    cerr << "t" << i << ":" << endl << this->t[i].t() << endl;
                    }
                for (int i=0; i<images.size(); i++)
                    if (images_processed.find(i)!=images_processed.end())
                        {
                        Mat C = - this->R[i].t()*this->t[i];
                        cerr << "C." << image_names[i] << ":" << endl << C.t() << endl;
                        }
                cerr << "K:" << this->K << endl << endl;
                
                } // if (flag_do_bundle)
            
            } // pose estimation
        
        
        // now do triangulation
        //
        cerr << "----------- triangulation ----------- " << endl;
        
        for (set<int>::iterator it=images_processed.begin(); it != images_processed.end(); ++it)
            {
            int processed_img_id = *it;
            std::pair<int,int> view_pair(processed_img_id, image_selected);
            cerr << "   pair (" << image_names[processed_img_id] << "," << image_names[image_selected] << ")" << endl;
            
            
            vector<DMatch> &matches = matches_pairs[view_pair];
            vector<Point2f> p1, p2;
            getAlignedPointsFromMatch(imgKeypts[processed_img_id], imgKeypts[image_selected],
                                      matches,
                                      p1, p2);
            
            // check if a match is already used for X3D
            //
            vector<uchar> inlier(matches.size(), 1);
            vector<int> already_used_for_3d(matches.size(),false);
            vector<int> used_3d_id(matches.size(),-1);
            
            for (int k=0; k<matches.size(); k++)
                for (int i=0; i<x3d.size(); i++)
                    if (x3d[i].ids[processed_img_id]==matches[k].queryIdx
                        || x3d[i].ids[image_selected]==matches[k].trainIdx)
                        {
                        already_used_for_3d[k]=true;
                        used_3d_id[k] = i;
                        //   cerr << " ! already reconstructed" << endl;
                        }
            cerr << "** new pairs found: " << cv::countNonZero(inlier) << endl;
            
            vector<Point3d> X3;
            triangulate(this->R[processed_img_id], this->t[processed_img_id],
                        this->R[image_selected], this->t[image_selected],
                        p1, p2, X3, &inlier);
            
            cerr << "   triangulation done with inliers " << cv::countNonZero(inlier) << endl;
            
            for (int i=0; i<inlier.size(); i++)
                if (inlier[i])
                    {
                    // if the train_id has a 3D, then
                    if (already_used_for_3d[i])
                        {
                        this->x3d[used_3d_id[i]].ids[processed_img_id] = matches[i].queryIdx;
                        this->x3d[used_3d_id[i]].ids[image_selected]   = matches[i].trainIdx;
                        }
                    else
                        {
                        P3D p3d;
                        p3d.X = X3[i];
                        p3d.ids.resize(images.size(), -1); // initialize
                        p3d.ids[processed_img_id] = matches[i].queryIdx;
                        p3d.ids[image_selected]   = matches[i].trainIdx;
                        this->x3d.push_back(p3d);
                        }
                    }
            cv::Mat img_done = images[processed_img_id].clone();
            cv::Mat img_new  = images[image_selected].clone();
            for (int i=0; i<inlier.size(); i++)
                if (inlier[i])
                    {
                    cv::Scalar color = getRandomColor();
                    cv::circle(img_done, p1[i], 7, color, 2);
                    cv::circle(img_done, p1[i], 2, color);
                    cv::circle(img_new, p2[i], 7, color, 2);
                    cv::circle(img_new, p2[i], 2, color);
                    }
            imshow("img_done", img_done);
            imshow("img_new", img_new);
            cerr << "------------------------------------------------------------------ " << endl;
            //cv::waitKey();
            }
        
        images_processed.insert(image_selected);
        
        // ba after registration of new 3D points
        //
        this->ba();
        cerr << "-------------------------------------------" << endl;
        cerr << "  ** After BA " << endl;
        printReprojectionError();
        
        //break;
        } // while
    
}