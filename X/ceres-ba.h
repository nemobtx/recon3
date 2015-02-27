//
//  ceres-ba.h
//  sfm
//
//  Created by Yongduek Seo on 2015. 2. 27..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#ifndef __sfm__ceres_ba__
#define __sfm__ceres_ba__

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;

#include "ceres/ceres.h"
#include "ceres/rotation.h"

struct Observation {
    int cam_id, str_id;
    vector<float> pt;
    float p[2];
};

// Read a Bundle Adjustment in the Large dataset.
struct Bundler
{
    struct BALProblem {
    public:
        BALProblem() {
            pp_bound_ = 10.;
        }
        
        ~BALProblem() {
            delete[] point_index_;
            delete[] camera_index_;
            delete[] observations_;
            delete[] parameters_;
            delete[] shared_internals_;
        }
        
        int num_observations()       const { return num_observations_;               }
        const double* observations() const { return observations_;                   }
        double* mutable_cameras()          { return parameters_;                     }
        double* mutable_points()           { return parameters_  + 6 * num_cameras_; }
        
        double* mutable_camera_for_observation(int i) {
            return mutable_cameras() + camera_index_[i] * 6;
        }
        double* mutable_point_for_observation(int i) {
            return mutable_points() + point_index_[i] * 3;
        }
        
        double* mutable_shared_internals() {
            return shared_internals_;
        }
        
        
        bool loadData (int nc, int np, int no,
                       vector<Observation>& obs,
                       vector<double>& motion,
                       double&focal, double&x0, double&y0)
        {
        num_cameras_ = nc;
        num_points_  = np;
        num_observations_ = no;
        
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        
        num_parameters_ = 6 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        
        num_shared_internals_ = 3; // focal length, x0, y0
        shared_internals_ = new double[num_shared_internals_];
        
        cerr << "reading the observations" << endl;
        for (int i = 0; i < num_observations_; ++i) {
            *(camera_index_ + i) = obs[i].cam_id;
            *(point_index_ + i) = obs[i].str_id;
            for (int j = 0; j < 2; ++j) {
                *(observations_ + 2*i + j) = obs[i].pt[j];
            }
        }
        
        cerr << "reading the motion + structure" << endl;
        for (int i = 0; i < num_parameters_; ++i) {
            *(parameters_ + i) = motion[i];
        }
        
        cerr << "reading the camera internals" << endl;
        shared_internals_[0] = focal;
        shared_internals_[1] = x0;
        shared_internals_[2] = y0;
        
        return true;
        
        }

        bool dumpData (vector<double>& motion,
                       double&focal, double&x0, double&y0)
        {
        for (int i = 0; i < num_parameters_; ++i)
            motion[i] = *(parameters_ + i);
        
        focal = shared_internals_[0];
        x0 = shared_internals_[1];
        y0 = shared_internals_[2];
        
        return true;
        }

        
        void SaveFile (string filename) {
            FILE *fptr = fopen (filename.c_str(), "w");
            fprintf(fptr, "%d %d %d\n", num_cameras_ , num_points_ , num_observations_ );
            for (int i = 0; i < num_observations_; ++i) {
                fprintf(fptr, "%d ", camera_index_ [i]);
                fprintf(fptr, "%d ", point_index_ [i]);
                for (int j = 0; j < 2; ++j)
                    fprintf(fptr, "%lf ", observations_ [ 2*i + j]);
                fprintf(fptr, "\n");
            }
            for (int i = 0; i < num_parameters_; ++i)
                fprintf(fptr, "%20.17le\n", parameters_ [i]);
            
            for (int i = 0; i < num_shared_internals_; i++)
                fprintf(fptr, "%lf\n", shared_internals_ [i]);
            
            fclose (fptr);
            cerr << "** result saved to the file: " << filename << endl;
            return;
        }

        
        bool LoadFile(const char* filename) {
            FILE* fptr = fopen(filename, "r");
            if (fptr == NULL) {
                return false;
            };
            
            cerr << "reading the header" << endl;
            FscanfOrDie(fptr, "%d", &num_cameras_);
            FscanfOrDie(fptr, "%d", &num_points_);
            FscanfOrDie(fptr, "%d", &num_observations_);
            
            
            point_index_ = new int[num_observations_];
            camera_index_ = new int[num_observations_];
            observations_ = new double[2 * num_observations_];
            
            num_parameters_ = 6 * num_cameras_ + 3 * num_points_;
            parameters_ = new double[num_parameters_];
            
            num_shared_internals_ = 3; // focal length, x0, y0
            shared_internals_ = new double[num_shared_internals_];
            
            cerr << "reading the observations" << endl;
            for (int i = 0; i < num_observations_; ++i) {
                FscanfOrDie(fptr, "%d", camera_index_ + i);
                FscanfOrDie(fptr, "%d", point_index_ + i);
                for (int j = 0; j < 2; ++j) {
                    FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
                }
            }
            
            cerr << "reading the motion + structure" << endl;
            for (int i = 0; i < num_parameters_; ++i) {
                FscanfOrDie(fptr, "%lf", parameters_ + i);
            }
            
            cerr << "reading the camera internals" << endl;
            for (int i = 0; i < num_shared_internals_; i++)
                FscanfOrDie(fptr, "%lf", shared_internals_ + i);
            //  FscanfOrDie(fptr, "%lf", &pp_bound_);
            
            return true;
        }
        /*
         num_cameras num_points num_observations
         camera_index  point_index   x y
         ...
         r1 r2 r3 t1 t2 t3
         ...
         X Y Z
         ...
         focal
         x0
         y0
         */
        
    private:
        template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
            int num_scanned = fscanf(fptr, format, value);
            if (num_scanned != 1) {
                LOG(FATAL) << "Invalid UW data file.";
            }
        }
        
        int num_cameras_;
        int num_points_;
        int num_observations_;
        int num_parameters_;
        int num_shared_internals_;
        
        int* point_index_;
        int* camera_index_;
        double* observations_;
        double* parameters_;
        double* shared_internals_;
        double  pp_bound_;
    }; // BAL_Problem

    // Templated pinhole camera model for used with Ceres.  The camera is
    // parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
    // focal length and 2 for radial distortion. The principal point is not modeled
    // (i.e. it is assumed be located at the image center).
    struct SnavelyReprojectionError {
        SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}
        
        template <typename T>
        bool operator()(const T* const cam_internal,
                        const T* const camera,
                        const T* const point,
                        T* residuals) const
        {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp =  p[0] / p[2];
        T yp =  p[1] / p[2];
        
        // Apply second and fourth order radial distortion.
        const T& focal = cam_internal[0];
        const T& x0 = cam_internal[1];
        const T& y0 = cam_internal[2];
        
        T predicted_x = focal * xp  + x0;
        T predicted_y = focal * yp  + y0;
        
        //T r2 = xp*xp + yp*yp;
        //T distortion = T(1.0) + r2  * (l1 + l2  * r2);
        
        // Compute final projected point position.
        //const T& focal = camera[6];
        //T predicted_x = focal * distortion * xp;
        //T predicted_y = focal * distortion * yp;
        
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        
        return true;
        }
        
        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double observed_x,
                                           const double observed_y)
        {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3/*internals*/, 6/*R,t*/, 3/*X*/>
                (
                 new SnavelyReprojectionError(observed_x, observed_y)
                 )
                );
        }
        
        double observed_x;
        double observed_y;
    };
    
    
    BALProblem bal_problem;

    int adjust()
    {
    const double* observations = bal_problem.observations();
    
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i)
        {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        
        ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 bal_problem.mutable_shared_internals(),
                                 bal_problem.mutable_camera_for_observation(i),
                                 bal_problem.mutable_point_for_observation(i)
                                 );
        }
    
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    
    //bal_problem.SaveFile ("ba.txt");
    return 0;
    }
    
    Bundler (int nc, int np, int no,
             vector<Observation>& obs,
             vector<double>& motion,
             double&focal, double&x0, double&y0)
    {
    bal_problem.loadData(nc, np, no, obs, motion, focal, x0, y0);
    adjust();
    bal_problem.dumpData(motion, focal, x0, y0);
    }
    ~Bundler()
    {
    }
}; // struct Bundler


#endif /* defined(__sfm__ceres_ba__) */
