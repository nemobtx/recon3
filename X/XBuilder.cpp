//
//  XBuilder.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include <dirent.h>
#include <unistd.h>

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

bool checkFileExist (const string filename)
{
    //  The access() function checks the accessibility of the file named by path
    // for the access permissions indicated by amode.  The value of amode is the
    // bitwise inclusive OR of the access permissions to be checked (R_OK for
    // read permission, W_OK for write permission and X_OK for execute/search
    // permission) or the existence test, F_OK.
    // All components of the pathname path are checked for access permissions (including F_OK).
    int res = access(filename.c_str(), R_OK | F_OK);
    if (res<0) {
        cerr << "** checkFileExist(" << filename << ") returned " << res << endl;
    }
    // res<0 means R_OK is not granted
    return 0==res;
}

void XBuilder::open_imgs_dir(string dir_name)
{
    if (dir_name.c_str() == NULL) {
        cerr << "invalid dir name: " << dir_name << endl;
        exit (1);
    }
    
    vector<string> files_;
    
    const string inputlist = dir_name + "/" + string("imagelist.txt");
    this->listFileExist = false;
    
    if (false == checkFileExist (inputlist))
        {
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
        else
            {
            cerr << ("Couldn't open the directory");
            return;
            }
        }
    else
        {
        ifstream fs(inputlist);
        while (!fs.eof())
            {
            string name;
            fs >> name;
            if (fs.eof()) break;
            cerr << "imae file: " << name << endl;
            files_.push_back(name);
            }
        fs.close();
        listFileExist = true;
        }
    
    // read all the images
    //
    for (unsigned int i=0; i<files_.size(); i++)
        {
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
    for (int i=0; i<image_names.size(); i++) {
        cerr << i << ": " << image_names[i] << endl;        
    }

    //
    //load calibration matrix
    //
    
    cv::FileStorage fs;
    std::string calibFile = dir_name + "/out_camera_data.yml";
    
    cerr << "  reading camera calibration data from <" << calibFile << ">" << endl;
    
//    cv::Mat cam_matrix, distortion_coeff;
    if(fs.open(calibFile,cv::FileStorage::READ)) {
        fs["camera_matrix"]>> this->K;
        fs["distortion_coefficients"]>> this->distortion_coeff;
    } else {
        std::cerr << "! calibration file does not exist. Using temporay values" << std::endl;
        //no calibration matrix file - mockup calibration
        cv::Size imgs_size = images[0].size();
        //double max_w_h = MAX(imgs_size.height,imgs_size.width);
        this->K = (cv::Mat_<double>(3,3) <<	800. ,	0	,		imgs_size.width/2.0,
                      0,		800.,	imgs_size.height/2.0,
                      0,			0,			1);
        distortion_coeff = cv::Mat_<double>::zeros(1,4);
    }
    
    cv::invert(K, Kinv); //get inverse of camera matrix
    
    std::cerr << "K=" << K << endl;
    std::cerr << "distortion_coeff= " << distortion_coeff << endl;

    return;
}

cv::Vec3b getRGB (const P3D p3d, const vector<vector<cv::KeyPoint> >& imgKeypts, const vector<cv::Mat>& images)
{
    vector<Vec3b> colors;
    for (int i=0; i<p3d.ids.size(); i++)
        {
        if (p3d.ids[i] >= 0)
            {
            cv::Vec3b rgb = images[i].at<cv::Vec3b>( imgKeypts[i][p3d.ids[i]].pt );
            colors.push_back(rgb);
            }
        }
    
    cv::Scalar mcolor = cv::mean(colors);
    return cv::Vec3b(mcolor[0],mcolor[1],mcolor[2]);
}


void XBuilder::fileSave(const std::string name)
{
    FILE *fp = fopen (name.c_str(), "w");
    if (fp==0) {
        cerr << "File open error: " << name << endl;
        exit (0);
    }
    
    double t = cv::getTickCount();
    fprintf(fp, "%lu\n", this->x3d.size());
    for (unsigned int i=0; i<this->x3d.size(); i++)
        {
        // get the RGB color value for the point
        cv::Vec3b rgbv(255,255,255);
        rgbv = getRGB (this->x3d[i], this->imgKeypts, this->images);
        
        cv::Point3d p3 = this->x3d[i].X;
        fprintf (fp, "%10.7f  %10.7f  %10.7f  %4d  %4d  %4d\n",
                 p3.x, p3.y, p3.z,
                 rgbv[0], rgbv[1], rgbv[2]);
        }
    fclose (fp);
    
    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    cout << "Done. (" << t <<"s)"<< endl;
    cout << "! file save to : " << name << endl;
}
