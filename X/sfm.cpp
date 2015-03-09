//
//  sfm.cpp
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 23..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#include "XBuilder.h"


void XBuilder::printReprojectionError()
{
    cerr << "XBuilder::printReprojectionError()" << endl;
//    for (int k=0; k<3; k++)
//        cerr << "R[" << k << "]:" << endl << this->R[k] << endl;
    
    double rms=0.;
    double mean=0.;
    int count=0;
    
    for (int i=0; i<x3d.size(); i++)
        {
        cv::Mat_<double> x3 (x3d[i].X);
        for (int k=0; k<x3d[i].ids.size(); k++)
            {
            if (x3d[i].ids[k] >= 0)
                {
                Mat_<double> p3 = R[k] * x3 + t[k];
                p3 /= p3(2);
                p3 = K * p3;
        
                cv::Point2f pt = this->imgKeypts[k][x3d[i].ids[k]].pt;

                double e1 = p3(0) - pt.x;
                double e2 = p3(1) - pt.y;

                //cerr << "R:" << R[k] << t[k] << "X:" << x3 << "pt" << pt << endl;

                double err2 = (e1*e1 + e2*e2);
                rms += err2;
                
                err2 = sqrt(err2);
                mean += err2;
                ++count;
                }
            }
        }
    rms = sqrt( rms / count );
    mean /= count;
    
    cerr << "! XBuilder:: Quality rms = " << rms << endl;
    cerr << "! XBuilder:: Quality mean L2 = " << mean << endl;
    cerr << "  ----" << endl;
}

static
cv::Mat_<double> makeRt(cv::Mat_<double> R1, cv::Mat_<double> t1)
{
    cv::Mat_<double> P1(3,4);
    for (int r=0; r<3; r++)
        {
        int c=0;
        for (; c<3; c++)
            P1(r,c) = R1(r,c);
        P1(r,c) = t1(r);
        }
    return P1;
}

vector<cv::Mat_<double> > getTrifocal (vector<Mat_<double> >& Parray, int ii, int jj, int kk)
{
    vector<cv::Mat_<double> > T(3);
    for (int i=0; i<3; i++) T[i] = cv::Mat_<double>(3,3);
    
    vector<cv::Mat_<double> > P;
    P.push_back(Parray[ii].clone());
    P.push_back(Parray[jj].clone());
    P.push_back(Parray[kk].clone());
    
    cv::Mat_<double> H = cv::Mat_<double>::eye(4,4);
    for (int r=0; r<3; r++) for (int c=0; c<4; c++) H(r,c) = P[0](r,c);
    cv::Mat_<double> Hinv = H.inv();
    for (int i=0; i<3; i++)
        P[i] = P[i] * Hinv;
    
    for (int i=0; i<3; i++)
        T[i] = P[1].col(i)*P[2].row(4).t() - P[1].row(4)*P[2].row(i).t();
    return T;
}

void XBuilder::sfm()
{
    // 0. gray scale conversion
    
    // 1. keypoint + descriptor
    
    vector<pair<int, std::pair<int,int> > > pairs = KeyPoint_FMatrix_Matching ();
    
    // 2. 2-view reconstruction
    Two_View_Reconstruction(pairs);
    
    fileSave("result3D-2View.txt");

    // reconstruction for other views
    //
    doPnP();
    
    {
    for (int n=0; n<this->images.size(); n++)
        {
        this->P[n] = this->K * makeRt(this->R[n], this->t[n]);
        }
    }
    
    set< pair<int,int> > usedObs;
    for (int i=0; i<x3d.size(); i++)
        for (int v=0; v<x3d[i].ids.size(); v++)
            if (x3d[i].ids[v] >= 0)
                usedObs.insert(make_pair(v, x3d[i].ids[v]));
    
    // find unmatched observations through the view if there is
    //
    for (int i=0; i<x3d.size(); i++)
        {
        for (int v=0; v<x3d[i].ids.size(); v++)
            if (x3d[i].ids[v]<0)
                {
                int viewId=v;
                // make projection
                cv::Mat_<double> proj = K * (R[viewId] * cv::Mat(x3d[i].X) + t[viewId]);
                proj /= proj(2);
                
                vector<int> candidate;
                for (int k=0; k<imgKeypts[viewId].size(); ++k)
                    if (usedObs.find(make_pair(viewId, k)) != usedObs.end()) // still not used
                        {
                        cv::Mat_<double> evec (proj(0)-imgKeypts[viewId][k].pt.x,
                                               proj(1)-imgKeypts[viewId][k].pt.y);
                        double dist = cv::norm(evec);
                        if (dist < 2.*f_ransac_threshold)
                            candidate.push_back(k);
                        }
                if (!candidate.empty())
                    {
                    // choose one of the least feature distance.
                    }
                }
        }
    
    // find new matches from scratch
    //
    {
    vector<int> perm;
    for (int i=0; i<images.size(); i++) perm.push_back(i);
    set< pair<int,pair<int,int> > > triples;
    do
        {
//        for (int i=0; i<perm.size(); i++)
//            cerr << perm[i] << ' ' ;
//        cerr << endl;
#define make_triple(a,b,c) (make_pair(a,make_pair(b,c)))
        triples.insert(make_triple(perm[0], perm[1], perm[2]));
        }
    while (std::next_permutation(perm.begin(), perm.end()));
    cerr << "Generated triples: " << triples.size() << endl;
    
    for (set< pair<int,pair<int,int> > >::iterator it=triples.begin(); it!=triples.end(); ++it)
        {
        vector<cv::Mat_<double> > T = getTrifocal (this->P,
                                                   it->first, it->second.first,it->second.second);
        
        }
    }
    
    // final ba
    //
    cerr << "-------------------------------------------" << endl;
    cerr << "** Final BA " << endl;
    cerr << "-------------------------------------------" << endl;
    this->ba();
    cerr << "-------------------------------------------" << endl;
    cerr << "  ** After BA " << endl;
    printReprojectionError();
    cerr << "-------------------------------------------" << endl;
    cerr << endl << "!!! SFM Finished !!!" << endl << endl;
}
