#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "CurveFitting.hpp"

using namespace std;
using namespace cv;

#if 0
double findLinePoint( const CurveFitting* fit_ptr )
{
    if( !fit_ptr )
        return numeric_limits<double>::infinity();
    double a = fit_ptr->param[0];
    double b = fit_ptr->param[1];
    double c = fit_ptr->param[2] - fit_ptr->computeValue( 2 * height );
    double left_discriminant = pow( b, 2 ) - 4 * a * c;
    double lane_point = numeric_limits<double>::infinity();
    if( left_discriminant > 0 )
    {
        double temp_x_1 = -b - sqrt( left_discriminant );
        double temp_x_2 = -b + sqrt( left_discriminant );
        if( abs( temp_x_1 ) < abs( temp_x_2 ) )
            lane_point = temp_x_1 / ( 2 * a );
        else
            lane_point = temp_x_2 / ( 2 * a );
    }
    return lane_point;
}
#endif

int masin()
{
    cout << "hello" << endl;

    string filename0 = "C:/Users/studml04/Pictures/Camera Roll/Capture03.png";
    Mat image = imread( filename0 );
    Mat grey_image;
    cvtColor( image, grey_image, COLOR_BGR2GRAY );
    Canny( grey_image, grey_image, 100, 200 );
    // imshow( "original", grey_image );
    // waitKey();

    Mat test = Mat::zeros( image.size(), image.type() );
    int height = image.rows / 2, width = image.cols;
    Mat region_of_interest = image( Rect( 0, height, width, height ) ).clone();
    Mat region_on_hood = Mat::zeros( height, width, image.type() );
    const int num_h = 4;
    Point points_h[1][num_h] = {Point( width * 0.1, height ), Point( width * 0.9, height ),
                                Point( width * 0.78, height * 0.85 ),
                                Point( width * 0.22, height * 0.85 )};
    const Point* ppt_h[1] = {points_h[0]};
    const int* npt_h = &num_h;
    fillPoly( region_of_interest, ppt_h, npt_h, 1, Scalar( 255 ) );
    imshow( "roi", region_of_interest );
    waitKey();

    // Point2f src[] = {Point2f( width * 0.1, height ), Point2f( width * 0.9, height ),
    //                    Point2f( width * 0.78, height * 0.85 ), Point2f( width * 0.22, height *
    //                    0.85 )};
    //   Point2f dst[] = {Point2f( width * 0.22, height * 1. ), Point2f( width * 0.78, height * 1.),
    //                    Point2f( width * 0.78, height * 0.85 ), Point2f( width * 0.22, height *
    //                    0.85 )};
    Point2f src[] = {Point2f( 0, 0.1 * height ), Point2f( width, 0.1 * height ),
                     Point2f( width, height ), Point2f( 0, height )};
    Point2f dst[] = {Point2f( 0, 0.1 * height ), Point2f( width, 0.1 * height ),
                     Point2f( width * 0.53, height ), Point2f( width * 0.47, height )};
    Mat transform = getPerspectiveTransform( src, dst );
    Mat result;
    // warpPerspective( region_of_interest, result, transform,
    // cv::Size( 2 * image.cols, 2 * image.rows ), cv::INTER_NEAREST );
    warpPerspective( region_of_interest, result, transform, region_of_interest.size(),
                     cv::INTER_LINEAR );
    // resize( result, result, Size( result.cols / 2, result.rows / 2 ) );
    imshow( "after inverse perspective", result );
    waitKey();

    Mat res_grey;
    cvtColor( result, res_grey, COLOR_BGR2GRAY );
    Canny( res_grey, res_grey, 100, 200 );
    imshow( "res_grey", res_grey );
    waitKey();
    vector<Vec4i> lines;
    vector<Point2f> left_lines, right_lines;
    HoughLinesP( res_grey, lines, 6, CV_PI / 60, 100, 50, 20 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i& l = lines[i];
        Point2f pt1, pt2;
        if( l[0] <= l[2] )
        {
            pt1 = Point2f( l[0], l[1] );
            pt2 = Point2f( l[2], l[3] );
        }
        else
        {
            pt1 = Point2f( l[2], l[3] );
            pt2 = Point2f( l[0], l[1] );
        }
        line( result, pt1, pt2, Scalar( 0, 255, 0 ), 3, LINE_AA );
    }
    imshow( "lines", result );
    waitKey();
    /*
    last left fitting parameter is [0.000344124, -0.675098, 480.416]
    last right fitting parameter is [-0.000350729, 0.99233, -87.3966]*/

    Vec3f param_left = {0.000344124, -0.675098, 480.416};
    Vec3f param_right = {-0.000350729, 0.99233, -87.3966};
}