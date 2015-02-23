//
//  XBuilder.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include <dirent.h>

#include "XBuilder.h"


static bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

static bool hasEndingLower (string const &fullString_, string const &_ending)
{
    string fullstring = fullString_, ending = _ending;
    transform(fullString_.begin(),fullString_.end(),fullstring.begin(),::tolower); // to lower
    return hasEnding(fullstring,ending);
}

void XBuilder::open_imgs_dir(string dir_name)
{
    if (dir_name.c_str() == NULL) {
        cerr << "invalid dir name: " << dir_name << endl;
        exit (1);
    }
    
    vector<string> files_;
    
//open a directory the POSIX way
    
    DIR *dp;
    struct dirent *ep;
    dp = opendir (dir_name.c_str());
    
    if (dp != NULL)
        {
        while ((ep = readdir (dp))) {
            if (ep->d_name[0] != '.')
                files_.push_back(ep->d_name);
        }
        
        (void) closedir (dp);
        }
    else {
        cerr << ("Couldn't open the directory");
        return;
    }
    
    // read all the images
    //
    for (unsigned int i=0; i<files_.size(); i++) {
        if (files_[i][0] == '.' ||
            !(hasEndingLower(files_[i],"jpg")||hasEndingLower(files_[i],"png")))
            {
            continue;
            }
        string imgfilename = string(dir_name).append("/").append(files_[i]);
        cerr << "trying to read: " << imgfilename << endl;
        cv::Mat m_ = cv::imread(imgfilename);
        if (m_.data == 0) {
            continue;
        }
        image_names.push_back(files_[i]);
        images.push_back(m_);
        cerr << "Image read: " << files_[i] << " of size " << images[i].cols << "x" << images[i].rows << endl;
    }
    
    if(images.size() == 0) {
        cerr << "can't get image files" << endl;
        exit (1);
    }
    cerr << "images loaded: " << endl;
    for (int i=0; i<image_names.size(); i++)
        cerr << i << ": " << image_names[i] << endl;

    return;
}

void XBuilder::fileSave(const std::string name)
{
    FILE *fp = fopen (name.c_str(), "w");
    if (fp==0) {
        cerr << "File open error: " << name << endl;
        exit (0);
    }
    
    double t = cv::getTickCount();
    fprintf(fp, "%lu\n", pointcloud.size());
    for (unsigned int i=0; i<pointcloud.size(); i++) {
        // get the RGB color value for the point
        cv::Vec3b rgbv(255,255,255);
        if (pointcloud_RGB.size() > i) {
            rgbv = pointcloud_RGB[i];
        }
        
        // 3D coordinates
        //        pclp.x = pointcloud[i].x;
        //        pclp.y = pointcloud[i].y;
        //        pclp.z = pointcloud[i].z;
        
        fprintf (fp, "%10.7f  %10.7f  %10.7f  %4d  %4d  %4d\n",
                 pointcloud[i].x, pointcloud[i].y, pointcloud[i].z,
                 rgbv[0], rgbv[1], rgbv[2]);
    }
    fclose (fp);
    
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    cout << "Done. (" << t <<"s)"<< endl;
    cout << "! file save to : " << name << endl;
}
