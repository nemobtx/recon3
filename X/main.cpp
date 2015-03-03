//
//  main.cpp
//  X
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include <iostream>

#include "XBuilder.h"
using namespace std;

XBuilder X;

int main(int argc, const char * argv[])
{
    chdir("../../../../../X");
    system("pwd");
    
    string default_dir = ".";
    
    const string image_dir = argv[1]? argv[1] : default_dir;
    
    X.open_imgs_dir(image_dir);
    
    //
    // perform SFM
    //
    X.sfm();
    
    // save the result
    // result3D.txt : 3D coordinates + RGB
    // features.txt : for each X_3D, its list of (image index, keypoint location, feature vector)
    // cameras.txt   : camera matrices
    X.fileSave("result3D.txt");

    return 0;
}
