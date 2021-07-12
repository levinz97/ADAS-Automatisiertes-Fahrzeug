#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "CurveFitting.hpp"

using namespace std;
using namespace cv;

// fitting y = a*x^2 + b*x + c
void CurveFitting ::computeResidual()
{
    for( auto point : points )
    {
        err = ( computeValue( point.x ) - point.y );
        computejacobi( point );
        max_x = max( max_x, point.x );
        min_x = min( min_x, point.x );
    }
    sumResidual += err;
}

void CurveFitting ::computejacobi( const Point2f& point )
{
    Mat_<float> temp_Jacobian;

	// if there are too few points, fit the line, otherwise parabola
    if( number_of_points > 4 )
        temp_Jacobian = Mat_<float>( {(float)pow( point.x, 2 ), point.x, 1.} );
    else 
		temp_Jacobian = Mat_<float>( {0, point.x, 1.} );

    Hessian += temp_Jacobian * temp_Jacobian.t();
    jacobians += temp_Jacobian * err;
}

void CurveFitting ::solveLinearsystem()
{
    Mat_<double> temp = -Hessian.inv() * jacobians;

    for( size_t i = 0; i < param_size; ++i )
    {
        delta_param[i] = temp[0][i];
    }
    // cout << delta_param << endl;  /////////////////////////////
}
void CurveFitting ::rollback_param()
{
    param -= delta_param;
}
bool CurveFitting ::isGoodStep()
{
    return abs( sumResidual ) < abs( last_sumResidual );
}
void CurveFitting ::solve( int iteration, int max_false_cnt )
{
    int iter = 0;
    int max_false_step = 3;
    while( max_false_step && iter < iteration )
    {
        // int false_cnt = -1;
        // do
        //{
        // rollback_param();
        // false_cnt++;
        sumResidual = 0;
        jacobians = Mat::zeros( param_size, 1, CV_32FC1 );
        Hessian = Mat::eye( param_size, param_size, CV_32FC1 );
        computeResidual();
#ifdef DEBUG_MODE
        cout << sumResidual << endl;
#endif
        solveLinearsystem();
        param += delta_param;
        //} while( !isGoodStep() && false_cnt < max_false_cnt);
        delta_param = 0;
        max_false_step -= !isGoodStep();
        last_sumResidual = sumResidual;
        iter++;
    }
}

float CurveFitting ::computeValue( int x ) const
{
    float value = 0;
    for( size_t i = 0; i < param_size; ++i )
    {
        value += pow( x, param_size - i - 1 ) * param[i];
    }

    return value;
}

void CurveFitting ::test()
{
    cout << "-------------------test begins------------------" << endl;
    vector<float> point{1, 2};
    int param = 3;
    Mat Hessian = Mat::eye( param, param, CV_32FC1 );
    auto jacobi = Mat_<float>( {point[0] * point[0], point[1], 3} );
    Mat_<double> temp = jacobi * jacobi.t();
    Vec3d param_;
    param_[0] = temp[0][0];
    // cout << delta_param << endl;

    vector<Point2f> points = {Vec2f{0, 0}, Vec2f{1, 1}, Vec2f{2, 4}, Vec2f{3, 9}, Vec2f{2, 4}};
    CurveFitting fitting( points );
    // cout << fitting.computejacobi( points[1] ) << endl;
    fitting.solve( 30 );

    cout << "----------------------result---------------" << endl;
    cout << fitting.param << endl;
    float res = 0;
    for( auto item : points )
    {
        cout << fitting.computeValue( item.x ) << endl;
        res += ( fitting.computeValue( item.x ) - item.y );
    }
    cout << "final residual is " << res << endl;

    cout << "=====================test end=====================" << endl;

    // Vec3f test = {1, 1, 1};
    //// test = 1;
    // cout << test << endl;
}