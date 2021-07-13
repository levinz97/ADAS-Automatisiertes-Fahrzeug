#pragma once

#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class CurveFitting
{
public:
    CurveFitting( std::vector<cv::Point2f> fitting_points, cv::Vec3d init_param = {0, 0, 0},
                  int param_num = 3 )
        : points( fitting_points ),
          param( init_param ),
          param_size( param_num ),
          number_of_points( fitting_points.size() )
    {
        if( number_of_points < 5 )
            init_param = 0;
        jacobians = cv::Mat::zeros( param_size, 1, CV_32FC1 );
        Hessian = cv::Mat::eye( param_size, param_size, CV_32FC1 );
        max_x = -1;
        min_x = 1e3;
    }
    void computeResidual();

    void computejacobi( const cv::Point2f& point );

    void solveLinearsystem();

    void rollback_param();

    bool isGoodStep();

    void solve( int iteration, int max_false_cnt = 0 );

    float computeValue( int x ) const;

    static void test();
    float max_x, min_x;  // used to plot the lane on screen
    cv::Vec3f param, delta_param;

	const std::vector<cv::Point2f> points;
    int param_size, number_of_points;
    double err, sumResidual, last_sumResidual;
    cv::Mat_<float> jacobians;
    cv::Mat_<float> Hessian;
};
