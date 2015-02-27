//
//  two_view_reconst.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 25..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"

inline bool sort_descending(pair<int,pair<int,int> > a, pair<int,pair<int,int> > b)
{ return a.first > b.first; }

inline bool sort_ascending(pair<int,pair<int,int> > a, pair<int,pair<int,int> > b)
{ return a.first < b.first; }


void XBuilder::Two_View_Reconstruction (vector<pair<int, std::pair<int,int> > >& pairs)
{
    std::sort(pairs.begin(), pairs.end(), sort_ascending);
    cerr << "hfratios ---- " << endl;
    for (int i=0; i<pairs.size(); i++)
        {
        cerr << pairs[i].first << endl;
        }
    cerr << "-" << endl;
    
    bool next_flag = true;
    int next_pair=0;
    do // find the most probable pair
        {
        int hfratio = pairs[next_pair].first;
        pair<int,int> min_pair = pairs[next_pair].second;
        ++next_pair;
        
        if (hfratio > 80) continue; // the motion of thew view pair is similar to rotaiton.
        
        cerr << endl;
        cerr << "! min_pair=" << image_names[min_pair.first] << ", " << image_names[min_pair.second] << endl;
        cerr << endl;
        
        //
        // 2-view reconstruction
        //
        vector<cv::Point2f> pt1, pt2;
        getAlignedPointsFromMatch(imgKeypts[min_pair.first], imgKeypts[min_pair.second], matches_pairs[min_pair], pt1, pt2);
        
        
#if 0
            {
            // re-do RANSAC for the two views
            vector<uchar> isInlier(pt1.size());
            cv::Mat F2 = cv::findFundamentalMat(pt1, pt2,
                                                FM_RANSAC,
                                                f_ransac_threshold/*pixel threshold*/,
                                                f_ransac_confidence,
                                                isInlier);
            
            if (cv::countNonZero(isInlier) < 100) continue;
            
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
            }
#endif
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
        
        if (pratio[0].first < 90/*percent*/)
            {
            cerr << "!! No positve reconstruction was obtained." << endl;
            continue;
            }
        
        
        
        cerr << " ------ " << endl << endl;
        cerr << "   good 2 view recontruction finished." << endl;
        cerr << endl << endl ;
        cerr << "pratio= " << pratio[0].first << endl;
        cerr << "between the pair: " << image_names[min_pair.first] << " , " << image_names[min_pair.second] << endl;
        cerr << "hfratio= " << hfratio << endl;
        cerr << "------" << endl;
        
        
        
        next_flag = false;
        
        int r_id = pratio[0].second.first;
        int t_id = pratio[0].second.second;
        
        cerr << "R* =" << Rs[r_id] << endl << "t* =" << ts[t_id] << endl;
        
        
        cerr << "--------- triangulation & outlier removal ----------" << endl;
        
        vector<uchar> inlier(pt1.size(),true);
        triangulate (R0, t0, Rs[r_id], ts[t_id], pt1, pt2, X3, &inlier);
        
        //
        // remove negatives, and those of large errors
        //
        vector<DMatch> matches;
        
        vector<double> err1 = reprojecionError (this->K, R0, t0, X3, pt1);
        vector<double> err2 = reprojecionError (this->K, Rs[r_id], ts[t_id], X3, pt2);
        
        for (int i=0; i<inlier.size(); i++)
            if (inlier[i] && (err1[i]<triangulation_err_th) && (err2[i]<triangulation_err_th))
                {
                matches.push_back(matches_pairs[min_pair][i]);
                }
        cerr << "! final matches result have " << matches.size() << " from " << matches_pairs[min_pair].size() << endl;
        if (matches.size() < 10)
            {
            cerr << "***   2 View Reconstruction has too small number of correspondences" << matches.size() << endl;
            exit (0);
            }
        // update the matches
        //
        matches_pairs[min_pair] = matches;
        pt1.clear(), pt2.clear();
        getAlignedPointsFromMatch(imgKeypts[min_pair.first], imgKeypts[min_pair.second], matches_pairs[min_pair], pt1, pt2);
        
        // re-do triangulation, final!
        triangulate (R0, t0, Rs[r_id], ts[t_id], pt1, pt2, X3);
        
        cerr << "**** 2 view finished ****" << endl;
        cerr << "R0" << R0 << endl << t0 << endl << "P1" << Rs[r_id] << ts[t_id] << endl;
        for (int ii=0; ii<3; ii++)
            cerr << "X:" << X3[ii] << endl << "p1:" << pt1[ii] << endl << "p2:" << pt2[ii] << endl << "--" << endl;
        cerr << "++++++++++++++++++++++" << endl;
        // 2-view finished -----------------------------------------------------------
        
        // allocation
        this->x3d.resize(X3.size());
        
        // record
        for (int m=0; m<X3.size(); m++)
            {
            this->x3d[m].X = X3[m];
            this->x3d[m].ids.resize(this->image_names.size(), -1);
            this->x3d[m].ids[min_pair.first] =matches_pairs[min_pair][m].queryIdx;
            this->x3d[m].ids[min_pair.second]=matches_pairs[min_pair][m].trainIdx;
            }
        
        this->R.resize(images.size());
        this->t.resize(images.size());
        this->R[min_pair.first] = R0;
        this->t[min_pair.first] = t0;
        this->R[min_pair.second] = Rs[r_id];
        this->t[min_pair.second] = ts[t_id];
        
        this->images_processed.resize(images.size(), false);
        this->images_processed[min_pair.first] = true;
        this->images_processed[min_pair.second] = true;
        
        /***
        for (int i=0; i<10; i++)
            cerr << "X:" << X3[i] << endl;
        for (int i=0; i<this->R.size(); i++)
            cerr << "R:"<<this->R[i] << endl << "t:" << this->t[i] << endl;
        ***/
        // the rest of the cams are packed with default; just in case ...
        for (int i=0; i<images.size(); i++)
            if (!(i==min_pair.first || i==min_pair.second))
                {
                this->R[i] = R0.clone();
                this->t[i] = t0.clone();
                }
        
        }
    while (next_flag==true);
    
    cerr << "-------------------------------------------" << endl;
    cerr << "  Two view reconstruction result before BA " << endl;
    printReprojectionError();

    // bundle-adjustment for the two views
    //
    this->ba ();

    cerr << "-------------------------------------------" << endl;
    cerr << "  Two view reconstruction result after BA " << endl;
    printReprojectionError();

    return;
}
// EOF //