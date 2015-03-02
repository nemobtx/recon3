//
//  ceres-pose.h
//  sfm
//
//  Created by Yongduek Seo on 2015. 3. 2..
//  Copyright (c) 2015년 Yongduek Seo. All rights reserved.
//

#ifndef __sfm__ceres_pose__
#define __sfm__ceres_pose__

#include <vector>
using namespace std;

#include "ceres/ceres.h"
#include "ceres/rotation.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct ObsPose {
    vector<double> p;
    vector<double> X;
};

struct CeresPose
{
    
    struct PoseProblem {
        double focal, x0, y0;
        double rtvec[6];
        double *observations_;
        int    num_obs;
        
        ~PoseProblem() {
            delete[] observations_;
        }
        
        int num_observations()       const { return num_obs; }
        const double* observations() const { return observations_; }
        
        void dump(vector<double>&rvec_, vector<double>&tvec_) {
            for (int k=0; k<rvec_.size(); k++)
                rvec_[k] = this->rtvec[k];
            for (int k=0; k<tvec_.size(); k++)
                tvec_[k] = this->rtvec[k+3];
        }
        
        void loadData(vector<ObsPose>& obs, vector<double>&rvec_, vector<double>& tvec_,
                      double focal_, double x0_, double y0_) {
            this->focal = focal_;
            this->x0 = x0_;
            this->y0 = y0_;

            for (int k=0; k<rvec_.size(); k++)
                this->rtvec[k] = rvec_[k];
            for (int k=0; k<tvec_.size(); k++)
                this->rtvec[k+3] = tvec_[k];
            
            this->observations_ = new double [5*(num_obs=(int)obs.size())];
            for (int i=0; i<num_obs; i++) {
                this->observations_[5*i+0] = obs[i].p[0];
                this->observations_[5*i+1] = obs[i].p[1];
                this->observations_[5*i+2] = obs[i].X[0];
                this->observations_[5*i+3] = obs[i].X[1];
                this->observations_[5*i+4] = obs[i].X[2];
            }
        }
    }; // Pose Problem

    PoseProblem poseProblem;
    
    struct ReprojectionError {
        const double observed_x;
        const double observed_y;
        const double U_;
        const double V_;
        const double W_;
        const double focal_, x0_, y0_;
        
        ReprojectionError(double x, double y,
                          double U, double V, double W,
                          double focal, double x0, double y0)
        : observed_x(x), observed_y(y), U_(U), V_(V), W_(W), focal_(focal), x0_(x0), y0_(y0) {}
        
        template<typename T>
        bool operator() (const T* const rtvec, T* residuals) const {
            T p[3];
            T point[3];
            
            point[0]= T(U_);
            point[1]= T(V_);
            point[2]= T(W_);
            
            ceres::AngleAxisRotatePoint(rtvec, point, p);
            p[0] += rtvec[3];
            p[1] += rtvec[4];
            p[2] += rtvec[5];
            
            T predicted_x = T(focal_ * p[0]/p[2] + x0_);
            T predicted_y = T(focal_ * p[1]/p[2] + y0_);
            
            residuals[0] = predicted_x - T(observed_x);
            residuals[1] = predicted_y - T(observed_y);
            
            return true;
        }
        
        static ceres::CostFunction* Create (const double x, const double y,
                                            const double u, const double v, const double w,
                                            const double focal, const double x0, const double y0) {
            return (new ceres::AutoDiffCostFunction<ReprojectionError,2,6>
                    (
                    new ReprojectionError(x,y,u,v,w,focal,x0,y0)
                     )
                    );
        }
    };

    int adjust () {
        const double* observations = poseProblem.observations();
        
        // Create residuals for each observation in the bundle adjustment problem. The
        // parameters for cameras and points are added automatically.
        ceres::Problem problem;
        for (int i = 0; i < poseProblem.num_observations(); ++i)
            {
            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            
            ceres::CostFunction* cost_function =
                   ReprojectionError::Create(observations[5 * i + 0],
                                             observations[5 * i + 1],
                                             observations[5 * i + 2],
                                             observations[5 * i + 3],
                                             observations[5 * i + 4],
                                             poseProblem.focal, poseProblem.x0, poseProblem.y0
                                             );
            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     poseProblem.rtvec
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
        
        return 0;
    }//adjust

    CeresPose (vector<ObsPose>& obs,
               vector<double>&rvec, vector<double>&tvec,
               double focal, double x0, double y0)
    {
    poseProblem.loadData(obs, rvec, tvec, focal, x0, y0);
    adjust();
    poseProblem.dump(rvec, tvec);
    }
};


#endif /* defined(__sfm__ceres_pose__) */
