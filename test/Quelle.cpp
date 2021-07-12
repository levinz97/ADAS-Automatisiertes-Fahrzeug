#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>

#include "CurveFitting.hpp"

#define DEBUG_MODE

//#define DEBUG_MODE

using namespace std;
using namespace cv;

struct Region_of_interest
{
    Mat region, image_;
    int width, height;
    explicit Region_of_interest( const Mat& image_ )
    {
        this->image_ = image_.clone();
        Mat grey_image;
        cvtColor( image_, grey_image, COLOR_BGR2GRAY );
        width = grey_image.cols, height = grey_image.rows / 2;

        if( grey_image.channels() != 1 )
        {
            cout << "type not supported\n";
        }
        // bottom half of original picture
        region = Mat::zeros( height, width, grey_image.type() );
        Mat canny_output;
        grey_image( Rect( 0, height, width, height ) ).copyTo( canny_output );

        // canny edge detector, with 2 threshold 100, 200
        Canny( canny_output, canny_output, 100, 200 );
        // imshow( "canny_output", canny_output );
        // waitKey();

        // region of interest is a polygon
        ////----------------------------------------------------------------------------important_parameters-----------------------
        const int num = 7;
        Point points[1][num] = {Point( width * 3 / 8, 0 ),  Point( width / 2, 0 ),
                                Point( 0, height * 3 / 4 ), Point( 0, height ),
                                Point( width, height ),     Point( width, height * 2 / 4 ),
                                Point( width * 5 / 8, 0 )};
        const Point* ppt[1] = {points[0]};

        const int* npt = &num;
        // fill the triangle with maximal grey value 0xff = 0b11111111
        fillPoly( region, ppt, npt, 1, Scalar( 255 ) );
        // bitwise_and() preserve all the value from canny_output in triangle region
        bitwise_and( canny_output, region, region );

#ifdef DEBUG_MODE
        imshow( "region of interest", region );
        waitKey();
#endif
        Mat res_Hough;
        image_.copyTo( res_Hough );

        // vector<Vec2f> lines;
        //     HoughLines( region, lines, 3, CV_PI / 90, 200 );  // runs the actual detection
        //     draw_lines( lines, res_Hough );

        vector<Vec4i> lines;
        vector<Point2f> left_lines, right_lines;
        HoughLinesP( region, lines, 6, CV_PI / 60, 200, 40,
                     20 );  ////------------------extremly important parameters----------------//
        // cout << lines.size() << endl;
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i& l = lines[i];
            Point2f pt1, pt2;
            if( l[0] <= l[2] )
            {
                pt1 = Point2f( l[0], l[1] += height );
                pt2 = Point2f( l[2], l[3] += height );
            }
            else
            {
                pt1 = Point2f( l[2], l[3] += height );
                pt2 = Point2f( l[0], l[1] += height );
            }
            line( res_Hough, pt1, pt2, Scalar( 0, 0, 255 ), 3, LINE_AA );
            double slope = ( 1. * l[3] - l[1] ) / ( l[2] - l[0] );
            if( abs( slope ) <
                tan( CV_PI / 180 *
                     20 ) )  ////----------------------important parameters-----------------------
                continue;

            if( pt1.inside( Rect( 0, height * 1.2, width / 2, height * 0.8 ) ) )
            {
                cout << "left point detected = " << pt1 << endl;
                left_lines.push_back( pt1 );
                left_lines.push_back( pt2 );
            }
            else if( pt1.inside( Rect( width / 2, height * 1.2, width, height * 0.8 ) ) )
            {
                cout << "right point detected = " << pt1 << endl;
                right_lines.push_back( pt1 );
                right_lines.push_back( pt2 );
            }
        }
#ifdef DEBUG_MODE
        imshow( "result of Hough transform", res_Hough );
        waitKey();
#endif
        generate_oneLine( left_lines, "left" );
        generate_oneLine( right_lines, "right" );
    }
    void generate_oneLine( vector<Point2f>& lines, string typeOfLines )
    {
        Scalar color = typeOfLines == "right" ? Scalar( 100, 100, 255 ) : Scalar( 255, 100, 0 );
        for( auto item : lines )
        {
            cout << "points is : " << item << endl;
        }
        if( lines.empty() )
        {
            cout << "no " << typeOfLines << " line detected!" << endl;
            return;
        }
        for( auto point : lines )
        {
            circle( image_, point, 10, color, 3 );
        }
#ifdef DEBUG_MODE
        imshow( typeOfLines + " points", image_ );
        waitKey();
#endif
        CurveFitting curve_fitting( lines );
        curve_fitting.solve( 30 );
        cout << curve_fitting.param << endl;

        draw_polynomial( curve_fitting, color );
    }

    void draw_polynomial( const CurveFitting& fit, Scalar color )
    {
        float start_point_x = 0, end_point = width;
        vector<Point2f> curvePoints;

        for( float x = start_point_x; x <= end_point; ++x )
        {
            float y = fit.computeValue( x );
            curvePoints.push_back( Point2f{x, y} );
        }
        Mat curve( curvePoints, true );
        curve.convertTo( curve, CV_32S );
        polylines( image_, curve, false, color, 2 );

        imshow( "final result", image_ );
        waitKey();
    }

    //////////////////////////////////////////////////////////////////////////////////
    void draw_polynomial()
    {
        Mat img( 350, 300, CV_8UC1, Scalar( 0 ) );
        Mat img2 = img.clone();

        float start_point_x = 20;
        float end_point_x = 120;
        vector<Point2f> curvePoints;

        // Define the curve through equation. In this example, a simple parabola
        for( float x = start_point_x; x <= end_point_x; x += 1 )
        {
            float y = 0.0425 * x * x - 6.25 * x + 258;
            Point2f new_point = Point2f( 2 * x, 2 * y );  // resized to better visualize
            curvePoints.push_back( new_point );           // add point to vector/list
        }

        // Option 1: use polylines
        Mat curve( curvePoints, true );
        curve.convertTo( curve, CV_32S );  // adapt type for polylines
        polylines( img, curve, false, Scalar( 255 ), 2, LINE_AA );

        // Option 2: use line with each pair of consecutives points
        for( int i = 0; i < curvePoints.size() - 1; i++ )
        {
            line( img2, curvePoints[i], curvePoints[i + 1], Scalar( 255 ), 2, LINE_AA );
        }

        imshow( "Curve 1 - polylines", img );
        imshow( "Curve 2 - line", img2 );
        waitKey();
    }

#if 0
    void draw_lines( vector<Vec2f>& lines, const Mat& res_Hough, int thickness = 3 ) const
    {
        if( lines.empty() )
        {
            cout << "no lines detected!" << endl;
            return;
        }
        if( res_Hough.empty() )
        {
            cout << "no image input" << endl;
            return;
        }
        // Draw the lines
        Point pt1( 0, 0 ), pt2( 0, 0 );
        int left_nums = 0;
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            if( rho > 0 )
            {
                left_nums++;
                double a = cos( theta ), b = sin( theta );
                double x0 = a * rho, y0 = b * rho;
                pt1.x += cvRound( x0 + 1000 * ( -b ) );
                pt1.y += cvRound( y0 + 1000 * ( a ) + height );
                pt2.x += cvRound( x0 - 1000 * ( -b ) );
                pt2.y += cvRound( y0 - 1000 * ( a ) + height );
            }
        }
        pt1.x /= left_nums;
        pt1.y /= left_nums;
        pt2.x /= left_nums;
        pt2.y /= left_nums;
        line( res_Hough, pt1, pt2, Scalar( 255 ), thickness, LINE_AA );
        imshow( "result of Hough transform", res_Hough );
        waitKey();
    }
#endif
};

void test()
{
    cout << "hello!" << endl;
    string filename0 = "C:/Users/studml04/Pictures/Camera Roll/test_lane.png";
    string filename1 = "C:/Users/studml04/Pictures/Camera Roll/curve01.png";
    string filename2 = "C:/Users/studml04/Pictures/Camera Roll/curve02.jpg";
    string filename3 = "C:/Users/studml04/Pictures/Camera Roll/curve03.png";

    string filename4 = "C:/Users/studml04/Pictures/Camera Roll/capture0.png";
    string filename5 = "C:/Users/studml04/Pictures/Camera Roll/capture01.png";
    Mat testImage = imread( filename5 );
    Region_of_interest roi( testImage );
#ifdef DEBUG_MODE
    cout << "this is debug mode!" << endl;
#endif

    // CurveFitting::test();
    // imshow( filename, testImage );
    // waitKey();
}