//
//  sfm.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"

void XBuilder::sfm()
{
    // 1. keypoint + descriptor
    cv::FeatureDetector *detector = cv::FeatureDetector::create("PyramidFAST");
    cv::DescriptorExtractor * extractor = cv::DescriptorExtractor::create("ORB");
    
    std::vector<std::vector<cv::KeyPoint> > imgKeypts;
    std::vector<cv::Mat> descriptors;

    std::cout << " -------------------- extract feature points for all images -------------------\n";
    detector->detect(images, imgKeypts);
    extractor->compute(images, imgKeypts, descriptors);
    std::cout << " ------------------------------------- done -----------------------------------\n";

}
