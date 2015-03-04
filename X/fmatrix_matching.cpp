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

void symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
		                const std::vector<std::vector<cv::DMatch> >& matches2,
                  std::vector<cv::DMatch>& symMatches)
{
    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator1= matches1.begin();
         matchIterator1!= matches1.end(); ++matchIterator1)
        {
        if (matchIterator1->size() < 2) // ignore deleted matches
            continue;
        
        // for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2= matches2.begin();
             matchIterator2!= matches2.end(); ++matchIterator2)
            {
            if (matchIterator2->size() < 2) // ignore deleted matches
                continue;
            
            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx  &&
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
                {
                // add symmetrical match
                symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
                                                (*matchIterator1)[0].trainIdx,
                                                (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
                }
            }
        }
    return;
}

int ratioTest(std::vector<std::vector<cv::DMatch> >& matches)
{
    const double ratio = 0.66;
    int removed=0;
    
    // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator= matches.begin();
         matchIterator!= matches.end(); ++matchIterator)
        {
        // if 2 NN has been identified
        if (matchIterator->size() > 1)
            {
            // check distance ratio
            if ((*matchIterator)[0].distance/(double)((*matchIterator)[1].distance) > ratio)
                {
                matchIterator->clear(); // remove match
                removed++;
                }
            
            }
        else
            { // does not have 2 neighbours
                matchIterator->clear(); // remove match
                removed++;
            }
        }
    
    return removed;
}

// Identify good matches using RANSAC
// Return fundemental matrix
cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		           const std::vector<cv::KeyPoint>& keypoints1,
                   const std::vector<cv::KeyPoint>& keypoints2,
                   std::vector<cv::DMatch>& outMatches)
{
    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
         it!= matches.end(); ++it)
        {
        // Get the position of left keypoints
        float x= keypoints1[it->queryIdx].pt.x;
        float y= keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
        // Get the position of right keypoints
        x= keypoints2[it->trainIdx].pt.x;
        y= keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x,y));
    }
    
    // Compute F matrix using RANSAC
    std::vector<uchar> inliers(points1.size(),0);
    cv::Mat fundemental= cv::findFundamentalMat(
                                                cv::Mat(points1),cv::Mat(points2), // matching points
                                                CV_FM_RANSAC, // RANSAC method
                                                f_ransac_threshold,     // distance to epipolar line
                                                f_ransac_confidence,  // confidence probability
                                                inliers      // match status (inlier ou outlier)
                                                );
    // extract the surviving (inliers) matches
    std::vector<uchar>::const_iterator itIn= inliers.begin();
    std::vector<cv::DMatch>::const_iterator itM= matches.begin();
    // for all matches
    for ( ;itIn!= inliers.end(); ++itIn, ++itM)
        {
        if (*itIn)
            { // it is a valid match
                outMatches.push_back(*itM);
            }
        }
    
    std::cout << "Number of matched points (after cleaning): " << outMatches.size() << std::endl;
    
    if (1)
        {
        // The F matrix will be recomputed with all accepted matches
        
        // Convert keypoints into Point2f for final F computation
        points1.clear();
        points2.clear();
        
        for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();
             it!= outMatches.end(); ++it)
            {
            // Get the position of left keypoints
            float x= keypoints1[it->queryIdx].pt.x;
            float y= keypoints1[it->queryIdx].pt.y;
            points1.push_back(cv::Point2f(x,y));
            // Get the position of right keypoints
            x= keypoints2[it->trainIdx].pt.x;
            y= keypoints2[it->trainIdx].pt.y;
            points2.push_back(cv::Point2f(x,y));
            }
        
        // Compute 8-point F from all accepted matches
        fundemental= cv::findFundamentalMat(
                                            cv::Mat(points1),cv::Mat(points2), // matching points
                                            CV_FM_8POINT); // 8-point method
        }
    
    return fundemental;
}


std::vector<cv::DMatch>
flipMatches(const std::vector<cv::DMatch>& matches)
{
    std::vector<cv::DMatch> flip;
    for(int i=0;i<matches.size();i++) {
        flip.push_back(matches[i]);
        swap(flip.back().queryIdx,flip.back().trainIdx);
    }
    return flip;
}


std::vector<pair<int, std::pair<int,int> > >
XBuilder::KeyPoint_FMatrix_Matching ()
{
    cv::Ptr<cv::FeatureDetector>     detector = cv::FeatureDetector::create("PyramidFAST");
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("ORB");
    cv::BFMatcher                    feature_matcher (NORM_HAMMING);

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
    
    
    
    float min_hfratio = 1.0;
    std::vector<std::pair<int,std::pair<int,int> > > pairs;
    std::pair<int,int> min_pair;
    for (int i=0; i< imgKeypts.size()-1; i++)
        {
        for (int k=i+1; k<imgKeypts.size(); k++)
            {
            cv::Mat disp;
            //std::vector<cv::DMatch> matches_1to2;
            vector<vector<DMatch> > match12, match21;
            
            cerr << "! BF-Matching of " << image_names[i] << " and " << image_names[k] << endl;
            // feature matching
            //
            //feature_matcher.match (descriptors[i], descriptors[k], matches_1to2); //  BF-Matching
            
            feature_matcher.knnMatch(descriptors[i], descriptors[k], match12, 2);
            feature_matcher.knnMatch(descriptors[k], descriptors[i], match21, 2);
            
            std::cout << "Number of matched points 1->2: " << match12.size() << std::endl;
            std::cout << "Number of matched points 2->1: " << match21.size() << std::endl;

            // ratio test
            //
            int removed= ratioTest(match12);
            std::cout << "Number of matched points 1->2 (ratio test) : " << match12.size()-removed << std::endl;
            removed= ratioTest(match21);
            std::cout << "Number of matched points 1->2 (ratio test) : " << match21.size()-removed << std::endl;

            // symmetry test
            //
            std::vector<cv::DMatch> symMatches;
            symmetryTest(match12,match21,symMatches);
            
            std::cout << "Number of matched points (symmetry test): " << symMatches.size() << std::endl;

            cv::drawMatches(images[i], imgKeypts[i], images[k], imgKeypts[k], symMatches, disp);
            imshow("view", disp);
            cv::waitKey(1000);
            
            // ransac test
            //
            std::vector<cv::DMatch> matches_F;
            cv::Mat F = ransacTest(symMatches, imgKeypts[i], imgKeypts[k], matches_F);
            
            cv::drawMatches(images[i], imgKeypts[i], images[k], imgKeypts[k], matches_F, disp);
            imshow("view", disp);
            cv::waitKey(1000);
            
            
            // homography computation
            //
            std::vector<cv::Point2f> pt1, pt2;
            getAlignedPointsFromMatch(imgKeypts[i], imgKeypts[k], matches_F, pt1, pt2);
            
            vector<uchar> statusH;
            cv::Mat H = cv::findHomography(pt1, pt2,
                                           statusH,
                                           CV_RANSAC,
                                           f_ransac_threshold * .667
                                           ); //threshold from Snavely07
            
            // ratio = H / F
            //
            float hfratio =  cv::countNonZero(statusH) / (float)matches_F.size();
            cerr << " ratio (H/F) = " << hfratio << endl;
            if (hfratio < min_hfratio)
                {
                min_hfratio = hfratio;
                min_pair = std::make_pair(i,k);
                }
            
            if (listFileExist)
                pairs.push_back(make_pair(60+i+k, make_pair(i, k)));
            else
                pairs.push_back(make_pair(hfratio*100, make_pair(i, k)));
            
            // make the record of the matching
            this->matches_pairs[std::make_pair(i,k)] = matches_F;
            this->matches_pairs[std::make_pair(k,i)] = flipMatches(matches_F);
            this->mapF[std::make_pair(i,k)] = F;
            } // for(k
        } // for (i
    

    return pairs;
}


// EOF //