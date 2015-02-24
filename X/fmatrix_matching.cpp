//
//  fmatrix_matching.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 24..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
    ps.clear();
    for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
    kps.clear();
    for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}

void getAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgKeypts1,
                               const std::vector<cv::KeyPoint>& imgKeypts2,
                               const std::vector<cv::DMatch>& matches,
                               std::vector<cv::Point2f>& pt_set1,
                               std::vector<cv::Point2f>& pt_set2)
{
    for (unsigned int i=0; i<matches.size(); i++)
        {
        //		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
        assert(matches[i].queryIdx < imgKeypts1.size());
        pt_set1.push_back(imgKeypts1[matches[i].queryIdx].pt);
        assert(matches[i].trainIdx < imgKeypts2.size());
        pt_set2.push_back(imgKeypts2[matches[i].trainIdx].pt);
        }
    return;
}

void getAlignedKeyPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
                                  const std::vector<cv::KeyPoint>& imgpts2,
                                  const std::vector<cv::DMatch>& matches,
                                  std::vector<cv::KeyPoint>& pt_set1,
                                  std::vector<cv::KeyPoint>& pt_set2)
{
    for (unsigned int i=0; i<matches.size(); i++)
        {
        //		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
        assert(matches[i].queryIdx < imgpts1.size());
        pt_set1.push_back(imgpts1[matches[i].queryIdx]);
        assert(matches[i].trainIdx < imgpts2.size());
        pt_set2.push_back(imgpts2[matches[i].trainIdx]);
        }
    return;
}


std::pair<int,int> XBuilder::KeyPoint_FMatrix_Matching ()
{
    cv::Ptr<cv::FeatureDetector>     detector = cv::FeatureDetector::create("PyramidFAST");
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("ORB");
    
    //std::vector<std::vector<cv::KeyPoint> > imgKeypts(images.size());
    this->imgKeypts.resize(images.size());
    
    std::vector<cv::Mat> descriptors(images.size());
    
    std::cout << " - extract feature points for all images -\n";
    for (int i=0; i<this->images.size(); i++) {
        cerr << this->image_names[i] << endl;
        cv::Mat gray;
        if (images[i].channels()>1)
            cv::cvtColor(images[i], gray, CV_BGR2GRAY);
        else
            gray = images[i];
        
        cv::imshow("view", gray);
        cv::waitKey(1000);
        detector->detect(gray, imgKeypts[i]);
        extractor->compute(gray, imgKeypts[i], descriptors[i]);
    }
    std::cout << " - done -\n";
    
    
    BFMatcher feature_matcher (NORM_HAMMING,true);
    float min_hfratio = 1.0;
    std::pair<int,int> min_pair;
    for (int i=0; i< imgKeypts.size()-1; i++)
        {
        for (int k=i+1; k<imgKeypts.size(); k++)
            {
            cv::Mat disp;
            std::vector<cv::DMatch> matches_1to2;
            
            cerr << "! BF-Matching of " << image_names[i] << " and " << image_names[k] << endl;
            // feature matching
            //
            feature_matcher.match (descriptors[i], descriptors[k], matches_1to2); //  BF-Matching
            
            cv::drawMatches(images[i], imgKeypts[i], images[k], imgKeypts[k], matches_1to2, disp);
            imshow("view", disp);
            cv::waitKey(1000);
            // fmatrix_match (imgKeypts[i], imgKeypts[k], matches_1to2);
            //
            cerr << "! F-RANSAC Matching" << endl;
            std::vector<cv::KeyPoint> kpt1, kpt2; // index-aligned
            std::vector<cv::Point2f> pt1, pt2;
            getAlignedKeyPointsFromMatch(imgKeypts[i], imgKeypts[k], matches_1to2, kpt1, kpt2);
            KeyPointsToPoints(kpt1, pt1);
            KeyPointsToPoints(kpt2, pt2);
            
            vector<uchar> isInlier(pt1.size());
            cv::Mat F = cv::findFundamentalMat(pt1, pt2,
                                               FM_RANSAC,
                                               f_ransac_threshold/*pixel threshold*/,
                                               0.99,
                                               isInlier);
            
            kpt1.clear();
            kpt2.clear();
            std::vector<cv::DMatch> matches_F;
            for (unsigned a=0; a<isInlier.size(); a++)
                if (isInlier[a])
                    {
                    matches_F.push_back (matches_1to2[a]);
                    kpt1.push_back(imgKeypts[i][a]);      // re-collect keypoints (inliers)
                    kpt2.push_back(imgKeypts[k][a]);
                    }
            cerr << "! match result by F-RANSAC: " << matches_F.size() << " from " << matches_1to2.size() << endl;
            
            cv::drawMatches(images[i], imgKeypts[i], images[k], imgKeypts[k], matches_F, disp);
            imshow("view", disp);
            cv::waitKey(1000);
            
            vector<uchar> statusH;
            cv::Mat H = cv::findHomography(pt1, pt2,
                                           statusH,
                                           CV_RANSAC,
                                           f_ransac_threshold * .667
                                           ); //threshold from Snavely07
            
            float hfratio =  cv::countNonZero(statusH) / (float)cv::countNonZero(isInlier);
            cerr << " ratio (H/F) = " << hfratio << endl;
            if (hfratio < min_hfratio)
                {
                min_hfratio = hfratio;
                min_pair = std::make_pair(i,k);
                }
            
            // make the record of the matching
            this->matches_pairs[std::make_pair(i,k)] = matches_F;
            this->mapF[std::make_pair(i,k)] = F;
            
            } // for(k
        } // for (i
    

    return min_pair;
}


// EOF //