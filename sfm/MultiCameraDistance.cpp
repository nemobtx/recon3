/*****************************************************************************
*   ExploringSfMWithOpenCV
******************************************************************************
*   by Roy Shilkrot, 5th Dec 2012
*   http://www.morethantechnical.com/
******************************************************************************
*   Ch4 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#include "MultiCameraDistance.h"
#include "RichFeatureMatcher.h"
#include "OFFeatureMatcher.h"
using namespace std;

//c'tor
MultiCameraDistance::MultiCameraDistance(
	const std::vector<cv::Mat>& imgs_, 
	const std::vector<std::string>& imgs_names_, 
	const std::string& imgs_path_):
imgs_names(imgs_names_),features_matched(false),use_rich_features(true)//,use_gpu(true)
{		
	std::cout << "=========================== Load Images ===========================\n";
	//ensure images are CV_8UC3
	for (unsigned int i=0; i<imgs_.size(); i++) {
		imgs_orig.push_back(cv::Mat_<cv::Vec3b>());
		if (!imgs_[i].empty()) {
			if (imgs_[i].type() == CV_8UC1) {
				cvtColor(imgs_[i], imgs_orig[i], CV_GRAY2BGR);
			} else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
				imgs_[i].convertTo(imgs_orig[i],CV_8UC3,255.0);
			} else {
				imgs_[i].copyTo(imgs_orig[i]);
			}
		}
		
        imgs.push_back(cv::Mat());  // initially, imgs is empty.
        cvtColor(imgs_orig[i],imgs[i], CV_BGR2GRAY); // convert to GRAY
		
		imgpts.push_back(std::vector<cv::KeyPoint>());
//		imgpts_good.push_back(std::vector<cv::KeyPoint>());
		std::cout << ".";
	}
	std::cout << std::endl;
		
	//load calibration matrix
	cv::FileStorage fs;
    std::string calibFile = imgs_path_+ "/out_camera_data.yml";

    std::cerr << "-------------------------------------------------------" << endl;
    std::cerr << "File: MultiCameraDistance.cpp" << std::endl ;
    std::cerr << "  reading camera calibration data from <" << calibFile << ">" << std::endl;
    std::cerr << "-------------------------------------------------------" << endl;

	if(fs.open(calibFile,cv::FileStorage::READ)) {
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;
	} else {
        std::cerr << "! calibration file does not exist. Using temporay values" << std::endl;
		//no calibration matrix file - mockup calibration
		cv::Size imgs_size = imgs_[0].size();
        //double max_w_h = MAX(imgs_size.height,imgs_size.width);
		cam_matrix = (cv::Mat_<double>(3,3) <<	800. ,	0	,		imgs_size.width/2.0,
												0,		800.,	imgs_size.height/2.0,
												0,			0,			1);
		distortion_coeff = cv::Mat_<double>::zeros(1,4);
	}
	
	K = cam_matrix;
	invert(K, Kinv); //get inverse of camera matrix

	distortion_coeff.convertTo(distcoeff_32f,CV_32FC1);
	K.convertTo(K_32f,CV_32FC1);

    std::cerr << "K=" << K << endl;
    std::cerr << "distortion_coeff= " << distortion_coeff << endl;
}

void MultiCameraDistance::OnlyMatchFeatures(int strategy)
{
	if(features_matched) return;
	
	if (use_rich_features) {
        feature_matcher = new RichFeatureMatcher(imgs,imgpts); // keypoints, descriptors are all extracted here.
	} else {
		feature_matcher = new OFFeatureMatcher(false/*use_gpu*/,imgs,imgpts);
	}	

	if(strategy & STRATEGY_USE_OPTICAL_FLOW)
		use_rich_features = false;

    int loop1_top = (int)imgs.size() - 1;
    int loop2_top = (int)imgs.size();
    
    for (int frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++)
        {
        for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
            {
            std::cout << "------------ Match " << imgs_names[frame_num_i] << ","<<imgs_names[frame_num_j]<<" ------------\n";
            std::vector<cv::DMatch> matches_tmp;

            // feature matching
            //
            feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp); //  BF-Matching
            
            // save to matrix of matches
            matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp; // 	std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;
            std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
            matches_matrix[std::make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip;
            }
        }

	features_matched = true;
}

void MultiCameraDistance::GetRGBForPointCloud(
                                              const std::vector<CloudPoint>& _pcloud,
                                              std::vector<cv::Vec3b>& RGBforCloud
                                              )
{
    RGBforCloud.resize(_pcloud.size());
    for (unsigned int i=0; i<_pcloud.size(); i++)  // for each 3D point
        {
        unsigned int good_view = 0;
        std::vector<cv::Vec3b> point_colors;
        for(; good_view < imgs_orig.size(); good_view++) // for each view
            {
            if(_pcloud[i].imgpt_for_img[good_view] != -1) // if the view was used for the 3D point
                {
                int pt_idx = _pcloud[i].imgpt_for_img[good_view]; // get the index of the 2d point
                if(pt_idx >= imgpts[good_view].size()) // error case
                    {
                    std::cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << imgpts[good_view].size() << std::endl;
                    continue;
                    }
                cv::Point _pt = imgpts[good_view][pt_idx].pt; // get the point coordinate
                assert(good_view < imgs_orig.size()
                       && _pt.x < imgs_orig[good_view].cols
                       && _pt.y < imgs_orig[good_view].rows);
                
                point_colors.push_back(imgs_orig[good_view].at<cv::Vec3b>(_pt));
                
                //				std::stringstream ss; ss << "patch " << good_view;
                //				imshow_250x250(ss.str(), imgs_orig[good_view](cv::Range(_pt.y-10,_pt.y+10),cv::Range(_pt.x-10,_pt.x+10)));
                }
            }
        //		cv::waitKey(0);
        cv::Scalar res_color = cv::mean(point_colors);
        RGBforCloud[i] = (cv::Vec3b(res_color[0],res_color[1],res_color[2])); //bgr2rgb
        if(good_view == imgs.size()) //nothing found.. put red dot; this must not happen
            RGBforCloud.push_back(cv::Vec3b(255,0,0));
        }// end for
    
    return;
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------
// image index, keypoint, descriptor

struct myFeatureMap {
    int image_index;
    cv::Point2f pt2f;
    cv::Mat_<float> desc;
};
vector<myFeatureMap> fmapCloud;

void MultiCameraDistance::GetFeatureMapForPointCloud(
                                              const std::vector<CloudPoint>& _pcloud
                                              )
{
    fmapCloud.resize(_pcloud.size());
    for (unsigned int i=0; i<_pcloud.size(); i++)  // for each 3D point
        {
        unsigned int good_view = 0;
        std::vector<cv::Vec3b> point_colors;
        for(; good_view < imgs_orig.size(); good_view++) // for each view
            {
            if(_pcloud[i].imgpt_for_img[good_view] != -1) // if the view was used for the 3D point
                {
                int pt_idx = _pcloud[i].imgpt_for_img[good_view]; // get the index of the 2d point
                if(pt_idx >= imgpts[good_view].size()) // error case
                    {
                    std::cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << imgpts[good_view].size() << std::endl;
                    continue;
                    }
                cv::Point _pt = imgpts[good_view][pt_idx].pt; // get the point coordinate
                assert(good_view < imgs_orig.size()
                       && _pt.x < imgs_orig[good_view].cols
                       && _pt.y < imgs_orig[good_view].rows);
                
                point_colors.push_back(imgs_orig[good_view].at<cv::Vec3b>(_pt));
                
                //				std::stringstream ss; ss << "patch " << good_view;
                //				imshow_250x250(ss.str(), imgs_orig[good_view](cv::Range(_pt.y-10,_pt.y+10),cv::Range(_pt.x-10,_pt.x+10)));
                }
            }
        //		cv::waitKey(0);
//        cv::Scalar res_color = cv::mean(point_colors);
//        fmapCloud[i] = (cv::Vec3b(res_color[0],res_color[1],res_color[2])); //bgr2rgb
//        if(good_view == imgs.size()) //nothing found.. put red dot; this must not happen
//            fmapCloud.push_back(cv::Vec3b(255,0,0));
        }// end for
    
    return;
}

// EOF //