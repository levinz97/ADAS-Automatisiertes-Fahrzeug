#include <iostream>
#include <fstream>
#include <string>
#include <memory>
//#include <thread>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>

#include "CurveFitting.hpp"
#include "PidController.hpp"
#include "ObjectDetection.hpp"

#define DEBUG_MODE
//#define DEBUG_STEERING
#define DEBUG_ACC
//#define DRAW_POLYGON // used for visualization
#define ENABLE_OBJECT_DETECTION 1
#define PRINT_VALUE_ACC 1
#define PRINT_VALUE_STEERING 0

// using namespace std;
using std::cout;
using std::endl;
using std::make_shared;
using std::max;
using std::min;
using std::numeric_limits;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;
// using namespace cv;
using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Rect;
using cv::Scalar;
using cv::Size;
using cv::Vec3f;
using cv::Vec4i;

// file location of neural network
string weightPath = "Mydnn/v3/frozen_inference_graph.pb";
string configPath = "Mydnn/v3/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";

class LaneAssistant
{
public:
    /*steering pid controller based on speed
        Kp,		Ki,		Kd
    -0.005, -1e-4, -0.01
    -0.003, -0.00005, -0.005 speed > 50 km/h
    -0.006, -0.00005, -0.001 low speed*/
    LaneAssistant()
        : steeringController( -0.003, -0.000005, -0.005 ),
          speedController( -0.3, -5e-6, -5 ),
          distanceController( -0.07, -5e-7, -5 )
#if ENABLE_OBJECT_DETECTION
          ,
          object_detector( weightPath, configPath )
#endif
    {
        left_last_fparam = 0;
        right_last_fparam = 0;
        width = height = 0;
        last_left_max = last_right_min = 0;
        center_of_lane = 360;
        is_leftline_detected = is_rightline_detected = true;
        curr_time = last_time = 0.;
        throttle_input = 0.;
        min_distance = numeric_limits<double>::max();
        need_brake = false;
#if ENABLE_OBJECT_DETECTION
        objects_in_camera = {};
#endif
        send_steering_value = send_throttle_value = true;
        ego_velocity_ = 0;
        image_ = Mat::zeros( Size( 512, 720 ), CV_32F );
        // is_object_insight = false;
    }

    // do stuff with data
    // send results via socket
    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
        if( send_steering_value )
            setSteeringInput( socket );
        if( send_throttle_value )
            setThrottleInput( socket );
        return true;
    }

protected:
    void setThrottleInput( tronis::CircularMultiQueuedSocket& socket )
    {
        double set_min_dist = 10;   // in meter
        double set_max_speed = 50;  // in km/h
        // TODO: set_max_speed should = min(velocity_of_the_frontal_car, 50) if min_distance <
        // threshold.

        cout << "min_distance is " << min_distance << endl;
        cout << "ego_velocity is " << ego_velocity_ << endl;

        if( min_distance < set_min_dist && ego_velocity_ > 5 ||
            min_distance < set_min_dist + 5 && ego_velocity_ > 40 ||
            min_distance < set_min_dist + 10 && ego_velocity_ > 45 || need_brake )
        {
            // to close to the front car, need need_brake assist
            string prefix = "brake,";
            double brake_intensity = 1.;
            socket.send( tronis::SocketData( prefix + to_string( brake_intensity ) ) );
            cout << "time to brake!" << endl;
            // if car stops, clear the i_error in pidController.
            distanceController.setZero();
            return;
        }
        else if( min_distance > 50 )
        {
            // if distance to the front car is more than 50m, only control the speed
            cout << "clear to go " << endl;
            double speed_err = ego_velocity_ - set_max_speed;  // error in km/h
            speedController.UpdateErrorTerms( speed_err );
            throttle_input = speedController.OutputToActuator( 0.5, PRINT_VALUE_ACC );
            distanceController.setZero();
        }
        else
        {
            // there is a car ahead, controlled by both min_distance and set_max_speed
            double dist_err = set_min_dist - min_distance;
            if( dist_err > 0 )
                dist_err *= 10;
            distanceController.UpdateErrorTerms( dist_err );
            throttle_input = distanceController.OutputToActuator( 0.6, PRINT_VALUE_ACC );
            if( ego_velocity_ > set_max_speed )
            {
                double speed_err = ego_velocity_ - set_max_speed;  // error in km/h
                speedController.UpdateErrorTerms( speed_err );
                double temp_throttle = speedController.OutputToActuator( 0.5, PRINT_VALUE_ACC );

                throttle_input = min( temp_throttle, throttle_input );
            }
        }

#ifdef DEBUG_ACC
        cout << "throttle input is " << throttle_input << endl;
        // throttle_input = 1.;
#endif
        if( throttle_input > 1 )
            throttle_input = 1;
        else if( throttle_input < 0 && abs( ego_velocity_ ) < 1 )
            // if the car is still, prevent it from moving backwards
            throttle_input = 0;
        if( throttle_input < -1 )
            throttle_input = -1;
        string prefix = "throttle,";
        socket.send( tronis::SocketData( prefix + to_string( throttle_input ) ) );
    }

    void setSteeringInput( tronis::CircularMultiQueuedSocket& socket )
    {
        // cout << "width_of_image = " << width << endl;
        double err = 0.;
        if( abs( width / 2. - center_of_lane ) > 1e-2 )
            err = width / 2. - center_of_lane;
        steeringController.UpdateErrorTerms( err );
        double steering = steeringController.OutputToActuator( 0.5, PRINT_VALUE_STEERING );
        // cout << "the steering before " << steering << endl;
        // steering /= 100;
        if( steering > 1 )
            steering = 1;
        if( steering < -1 )
            steering = -1;
        // static int zigzag_cnt = 0;  // count the number of rapid change of directions
        // static bool last_direction_sign = steering > 0;
        // if( abs( steering ) > 0.1 )
        //{
        //    if( last_direction_sign != steering > 0 )
        //        zigzag_cnt++;
        //    else
        //        zigzag_cnt = 0;
        //    if( zigzag_cnt > 3 )
        //    {
        //        cout << "to rapid steering change detected! " << zigzag_cnt << '\n';
        //        steeringController.setZero();
        //        zigzag_cnt = 0;
        //    }
        //}
        // last_direction_sign = steering > 0;
        if( !is_leftline_detected && is_rightline_detected )
        {
            // TODO: steering based on curvature when only one line detected
            steering = -0.3;
            steeringController.setZero();
        }
        if( !is_rightline_detected && is_leftline_detected )
        {
            steering = 0.3;
            steeringController.setZero();
        }

        string prefix = "steering,";
        //  prevent under steering at rapid curve, only applied at high speed > 75 km/h
        // if( abs( steering ) < 0.15 )
        //{
        //    if( last_left_max > width * 0.45 && last_right_min > width * 0.7 )
        //    {
        //        cout << "steering before is " << steering << endl;
        //        cout << "understeering detected, increase the steering!" << endl;
        //        steering =
        //            max( steering, ( last_left_max + last_right_min - width ) / ( 1. * width ) );
        //        // steeringController.setZero();
        //        prefix = " right turn under steering,";
        //    }
        //    if( last_left_max < 0.3 * width && last_right_min < 0.55*width )
        //    {
        //        cout << "steering before is " << steering << endl;
        //        cout << "understeering detected, increase the steering!" << endl;
        //        steering = min(
        //            steering, ( last_left_max + last_right_min - width ) / ( 1. * width ) );
        //        // steeringController.setZero();
        //        prefix = " left turn under steering,";
        //    }
        //}

        socket.send( tronis::SocketData( prefix + to_string( steering ) ) );
        // if( int( curr_time ) % 100 == 0 )
        //    steeringController.setZero();
#ifdef DEBUG_STEERING
        cout << "center_of_lane = " << center_of_lane << endl;
        cout << "steering is " << steering << endl;
#endif
    }

protected:
    // lane detection
    string image_name_;
    Mat image_;
    int width, height;  // size of under half of image
    Vec3f left_last_fparam, right_last_fparam;

    // lane keeping
    bool send_steering_value;
    double center_of_lane;
    PidController steeringController;
    bool is_leftline_detected, is_rightline_detected;
    double curr_time, last_time;

    // adaptive cruise controll
    bool send_throttle_value;
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;
    double throttle_input;
    double min_distance;
    // vector<double> all_distance;
    PidController speedController;
    PidController distanceController;
    bool need_brake;

#if ENABLE_OBJECT_DETECTION
    // object detection from camera image
    vector<Rect> objects_in_camera;
    ObjectDetection object_detector;
    // bool is_object_insight;
#endif
    int last_left_max, last_right_min;  // used to plot a more stable lane

    // Function to detect lanes based on camera image
    void detectLanes()
    {
        //// only the under part of picture will be used for object detection
        //      int translation = static_cast<int>( 1.1 * height );
        //      Mat detection_mat =
        //          image_( Rect( 0, translation, width, height * 1.75 - translation ) ).clone();
        //      // detect object from image
        // thread t1( &ObjectDetection::detectObject, &object_detector, ref(detection_mat),
        // ref(objects_in_camera), translation );
        ////cout << image_.size << endl;

        Mat grey_image;
        cvtColor( image_, grey_image, cv::COLOR_BGR2GRAY );
        width = grey_image.cols, height = grey_image.rows / 2;

        // bottom half of original picture
        Mat region_of_interest = Mat::zeros( height, width, grey_image.type() );
        Mat canny_output;
        grey_image( Rect( 0, height, width, height ) ).copyTo( canny_output );
        // canny edge detector, with 2 threshold 100, 200
        Canny( canny_output, canny_output, 100, 200 );
        // imshow( "canny_output", canny_output );
        // waitKey();

        // region_of_interest is a polygon
        ////----------------------------------------------------------------------------extremly_important_parameters-----------------------------------------
        const int num = 7;
        Point points[1][num] = {Point( width / 2, 0 ),
                                Point( width * 0.4, height * 0.05 ),
                                Point( 0, height * 0.75 ),
                                Point( 0, height ),
                                Point( width, height ),
                                Point( width, height * 0.6 ),
                                Point( width * 0.61, height * 0.0025 )};
        // fill the triangle with maximal grey value 0xff = 0b11111111
        // const Point* ppt[1] = {points[0]};
        // const int* npt = &num;
        // fillPoly( region_of_interest, ppt, npt, 1, Scalar( 255 ) );
        const Point* ppt = points[0];
        fillConvexPoly( region_of_interest, ppt, num, Scalar( 255 ) );
        // bitwise_and() preserve all the value from canny_output in triangle region_of_interest
        bitwise_and( canny_output, region_of_interest, region_of_interest );
        canny_output.release();

        // remove the lines on the hood
        Mat region_on_hood = Mat::zeros( height, width, grey_image.type() );
        grey_image.release();
        const int num_h = 4;
        Point points_h[1][num_h] = {Point( width * 0.1, height ), Point( width * 0.9, height ),
                                    Point( width * 0.75, height * 0.7 ),
                                    Point( width / 4, height * 0.7 )};
        const Point* ppt_h[1] = {points_h[0]};
        const int* npt_h = &num_h;
        fillPoly( region_of_interest, ppt_h, npt_h, 1, Scalar( 0 ) );
        // bitwise_and( region_of_interest, region_on_hood, region_of_interest);

#ifdef DEBUG_MODE
        // imshow( "region of interest", region_of_interest );
        const int* npt = &num;
        const Point* pppt[1] = {points[0]};
        Mat test_roi = Mat::zeros( height, width, image_.type() );
        fillPoly( test_roi, pppt, npt, 1, Scalar( 255, 255, 255 ) );
        bitwise_and( image_( Rect( 0, height, width, height ) ), test_roi, test_roi );
        fillPoly( test_roi, ppt_h, npt_h, 1, Scalar( 0 ) );
        // showImage( "region of interest", region_of_interest );
        showImage( "region of interest", test_roi );
        Mat original = image_.clone();
        imshow( "original input", original );
        // waitKey();
#endif
        Mat res_Hough;
        image_.copyTo( res_Hough );

#if ENABLE_OBJECT_DETECTION
        object_detector.drawBoundingBox( image_, objects_in_camera );
#endif
        vector<Vec4i> lines;
        vector<Point2f> left_lines, right_lines;
        ////---------------------------------------------------------------------------extremly_important_parameters----------------//
        HoughLinesP( region_of_interest, lines, 6, CV_PI / 60, 200, 40, 20 );
        // cout << lines.size() << endl;

        // find the most left and right points of detected lane, used to eliminate outliers
        float left_min = width / 2, right_max = width / 2;
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
#ifdef DEBUG_MODE
            line( res_Hough, pt1, pt2, Scalar( 0, 0, 255 ), 3, cv::LINE_AA );
#endif
            double slope = ( 1. * l[3] - l[1] ) / ( l[2] - l[0] );
            ////-------------------------------------------------------------------------------------important_parameters------------------------------------///
            if( abs( slope ) < tan( CV_PI / 180 * 10 ) )
                continue;
            left_min = min( left_min, pt1.x );
            right_max = max( right_max, pt2.x );

            // t1.join();

#if ENABLE_OBJECT_DETECTION
            // object_detector.drawBoundingBox( image_, objects_in_camera );
            // remove the influence of other Objects for lane detection
            if( !objects_in_camera.empty() )
            {
                bool point_not_reliable = false;
                for( const Rect& rect : objects_in_camera )
                {
                    if( pt1.inside( rect ) || pt2.inside( rect ) )
                    {
                        point_not_reliable = true;
                        break;
                    }
                }
                if( point_not_reliable )
                    continue;
            }
#endif
            if( pt1.inside( Rect( 0, height * 1.2, width * 0.45, height * 0.8 ) ) )
            {
                // cout << "left point detected = " << pt1 << endl;
                left_lines.push_back( pt1 );
                if( !pt2.inside( Rect( width / 2., height, width / 2., height * 2 ) ) )
                {
                    left_lines.push_back( pt2 );
                }
            }
            else if( pt2.inside( Rect( width * 0.55, height * 1.2, width / 2., height * 2 ) ) )
            {
                // cout << "right point detected = " << pt1 << endl;
                right_lines.push_back( pt2 );
                if( !pt1.inside( Rect( 0, height, width / 2., height ) ) )
                {
                    right_lines.push_back( pt1 );
                }
            }
        }

        if( left_min > 0.2 * width )
            left_lines.clear();
        if( right_max < 0.8 * width )
            right_lines.clear();

#ifdef DEBUG_MODE
        // imshow( "result of Hough transform", res_Hough );
        // waitKey( 10 );
        showImage( "result of Hough transform", res_Hough );
#endif
        shared_ptr<CurveFitting> fit_L_ptr = generateOneLine( left_lines, "left" );
        shared_ptr<CurveFitting> fit_R_ptr = generateOneLine( right_lines, "right" );
        is_leftline_detected = ( fit_L_ptr != nullptr );
        is_rightline_detected = ( fit_R_ptr != nullptr );

#ifdef DRAW_POLYGON
        drawPolygon( fit_L_ptr.get(), fit_R_ptr.get() );
#endif
        double left_point = findLinePoint( fit_L_ptr.get(), "left" );
        double right_point = findLinePoint( fit_R_ptr.get(), "right" );
        // cout << "left_point " << left_point << "right_point " << right_point << endl;
        if( isfinite( left_point ) && isfinite( right_point ) )
            center_of_lane = ( left_point + right_point ) / 2.;
        // if the estimated center_of_lane not reliable just go straight
        if( center_of_lane <= 0 || center_of_lane >= width )
            center_of_lane = width / 2.;
        // viusalisation: draw a stick pointing the current direction
        line( image_, Point2f( center_of_lane, 1.7 * height ), Point2f( width / 2, 2 * height ),
              Scalar( 104, 55, 255 ), 3 );
    }

    /**
     * find the intersect point between lane and bottom line, used to compute the center of lane.
     * i.e., find x for y = ax^2 + bx + c where y == height of total image_
     * @param: pointer to the CurveFitting class
     * @param: the type of line: "right" or "left"
     * @return: the x value (width) in picture, where right/left line intersect with the y == height
     */
    double findLinePoint( const CurveFitting* fit_ptr, string Type_of_lines )
    {
        if( !fit_ptr )
            return numeric_limits<double>::infinity();

        double a = fit_ptr->param[0];
        double b = fit_ptr->param[1];
        double c = fit_ptr->param[2] - 1.7 * height;  // make some simple prediction
        double left_discriminant = pow( b, 2 ) - 4 * a * c;
        double lane_point = numeric_limits<double>::infinity();
        if( left_discriminant > 0 )
        {
            double temp_x_1 = -b - sqrt( left_discriminant );
            double temp_x_2 = -b + sqrt( left_discriminant );
            // printf( "a = %f , b = %f , c = %f", a, b, c );
            // cout << "temp_x_1 = " << temp_x_1 / ( 2 * a ) << " temp_x_2 = " << temp_x_2 / ( 2 * a
            // ) << endl;
            if( abs( temp_x_1 ) < abs( temp_x_2 ) )
                lane_point = temp_x_1 / ( 2 * a );
            else
                lane_point = temp_x_2 / ( 2 * a );
            if( Type_of_lines == "right" && lane_point < 0 )
                lane_point = temp_x_2 / ( 2 * a );
        }
        // cout << "lane point = " << lane_point << endl;
        return lane_point;
    }
    /**
     * generate the second order parabola from points
     * @param: the detected points from probabilistic Hough
     * @param: the type of line: "right" or "left"
     * @return: the shared_ptr of CurveFitting class representing the parabola
     */
    shared_ptr<CurveFitting> generateOneLine( vector<Point2f>& lines, string type_of_lines )
    {
        if( lines.empty() )
        {
            cout << "no " << type_of_lines << " line detected!" << endl;
            return nullptr;
        }
#ifdef DEBUG_MODE
        Scalar color = Scalar( 255, 20, 20 );
        color = type_of_lines == "right" ? Scalar( 100, 100, 255 ) : Scalar( 255, 100, 0 );
        for( auto point : lines )
        {
            circle( image_, point, 10, color, 2 );
        }
        // imshow( type_of_lines + " points", image_ );
        // waitKey();
#endif
        shared_ptr<CurveFitting> fit_ptr;
        if( type_of_lines == "left" )
        {
            fit_ptr = make_shared<CurveFitting>( lines, left_last_fparam );
            fit_ptr->solve( 10 );
            left_last_fparam = fit_ptr->param;
            // cout << "last left fitting parameter is " << left_last_fparam << endl;
            drawPolynomial( fit_ptr.get(), type_of_lines );
        }
        else
        {
            fit_ptr = make_shared<CurveFitting>( lines, right_last_fparam );
            fit_ptr->solve( 10 );
            right_last_fparam = fit_ptr->param;
            // cout << "last right fitting parameter is " << right_last_fparam << endl;
            drawPolynomial( fit_ptr.get(), type_of_lines );
        }
        return fit_ptr;
    }

    /**
     * draw the detected parabola(left and right lines) on image_
     * @param: pointer to the CurveFitting class
     * @param: type of lines
     */
    void drawPolynomial( const CurveFitting* fit_ptr, string type_of_lines )
    {
        if( !fit_ptr )
            return;
        // Scalar color = Scalar( 104, 55, 255, 10 );
        Scalar color = Scalar( 255, 20, 20 );
#ifdef DEBUG_MODE
        color = type_of_lines == "right" ? Scalar( 100, 100, 255 ) : Scalar( 255, 100, 0 );
        //        for( auto item : lines )
        //        {
        //            cout << "points is : " << item << endl;
        //        }
#endif
        float start_point_x, end_point_x;
        if( type_of_lines == "left" )
        {
            start_point_x = 0;
            end_point_x = last_left_max + 0.1 * ( fit_ptr->max_x - last_left_max );

#ifdef DEBUG_MODE
            end_point_x = fit_ptr->max_x;
#endif
            last_left_max = end_point_x;
        }
        else
        {
            start_point_x = last_right_min + 0.1 * ( fit_ptr->min_x - last_right_min );

#ifdef DEBUG_MODE
            start_point_x = fit_ptr->min_x;
#endif

            last_right_min = start_point_x;
            end_point_x = width;
        }
        vector<Point2f> curvePoints;
        for( float x = start_point_x; x <= end_point_x; x += 5 )
        {
            float y = fit_ptr->computeValue( x );
            curvePoints.push_back( Point2f{x, y} );
        }

        Mat curve( curvePoints, true );
        curve.convertTo( curve, CV_32S );
        polylines( image_, curve, false, color, 6 );

#ifdef DEBUG_MODE
        //        imshow( "final result", image_ );
        //        waitKey();
#endif
    }

    // draw the areas that have been enclosed by 2 lines, only used for visualisation.
    void drawPolygon( const CurveFitting* fitL, const CurveFitting* fitR )
    {
        if( !fitL || !fitR )
        {
            cout << "no fit_ptr ptr" << endl;
            return;
        }

        vector<Point> curvePoints;
        int start_point_x = 0, end_point_x = last_left_max + 0.05 * ( fitL->max_x - last_left_max );
        for( int x = start_point_x; x <= end_point_x; x += 10 )
        {
            int y = fitL->computeValue( x );
            curvePoints.push_back( Point{x, y} );
        }
        start_point_x = last_right_min + 0.05 * ( fitR->min_x - last_right_min );
        end_point_x = width;
        for( int x = start_point_x; x <= end_point_x; x += 10 )
        {
            int y = fitR->computeValue( x );
            curvePoints.push_back( Point{x, y} );
        }
        Mat background = Mat::zeros( image_.size(), image_.type() );
        fillConvexPoly( background, curvePoints, Scalar( 200, 180, 80 ) );
        addWeighted( image_, 0.7, background, 0.3, 0, image_ );
    }

    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = msg->Velocity * 3.6 * 1e-2;  // from cm/s to Km/h
#ifdef DEBUG_ACC
        // cout << "ego_location is " << ego_location_.ToString() << " \n ego_orientation is "
        //     << ego_orientation_.ToString() << "\n ego_velocity is " << ego_velocity_ << endl;
#endif
        return true;
    }

    bool processObject( tronis::BoxDataSub* sensorData )
    {
        size_t num_of_objects = sensorData->Objects.size();
        // process data from ObjectListsSensor
        min_distance = numeric_limits<double>::max();
        bool static_object_ahead = false;
        for( size_t i = 0; i < num_of_objects; i++ )
        {
            const tronis::ObjectSub& object = sensorData->Objects[i];

            tronis::LocationSub location = object.Pose.Location;
            tronis::QuaternionSub orientation = object.Pose.Quaternion;
            string actorName = object.ActorName.Value();
            float pos_x = location.X / 100;
            float pos_y = location.Y / 100;
            double dist = sqrt( pow( pos_x, 2 ) + pow( pos_y, 2 ) );
            float angle = atan( pos_y / pos_x );
            cout << actorName << endl;
            // if( actorName.find( "Hatchback" ) == string::npos )
            //    continue;

#ifdef DEBUG_ACC
            cout << actorName << " at \n";
            cout << object.Pose.Location.ToString() << "\n";
            cout << "angle is " << angle << "\n";
            cout << "distance is " << dist << endl;
#endif
            if( actorName.find( "SnappyRoad" ) != string::npos )
                continue;

            if( object.Type )
            {
                // for movable and animated object constrain the minimal distance
                if( abs( angle ) > CV_PI * 25 / 180. )
                    continue;
                if( abs( pos_y ) < 3 )
                {
                    min_distance = min( dist, min_distance );
                }
            }
            else
            {
                if( static_object_ahead )
                    continue;
                // for static object just need_brake
                if( dist < max( 2 * ego_velocity_ / 3.6, 10. ) && abs( pos_y ) < 3 )
                {
                    need_brake = true;
                    static_object_ahead = true;
                }
                else
                    need_brake = false;
            }
        }
        if( !static_object_ahead )
            need_brake = false;
        cout << "number of objects is " << num_of_objects << endl;
        if( num_of_objects == 0 )
        {
#if ENABLE_OBJECT_DETECTION
            objects_in_camera.clear();
#endif
            return false;
        }
#if ENABLE_OBJECT_DETECTION
        // only the under part of picture will be used for object detection
        int translation = static_cast<int>( 1.1 * height );
        if( num_of_objects >= 2 )
            translation = static_cast<int>( 0.9 * height );
        Mat detection_mat =
            image_( Rect( 0, translation, width, height * 1.75 - translation ) ).clone();
        // detect object from image
        // since the camera works at 60Hz, to reduce the computational effort, detect object every
        // max_cnt frames
        size_t max_cnt = 7;
        static size_t cnt = 0;
        // cout << "cnt is " << cnt << endl;
        if( cnt > max_cnt )
        {
            bool object_detected =
                object_detector.detectObject( detection_mat, objects_in_camera, translation );
            if( !object_detected )
                cout << "no object detected in image" << endl;
            cnt = 0;
        }
        else
        {
            ++cnt;
        }
        // object_detector.drawBoundingBox( image_, objects_in_camera );
#endif
        return true;
    }

    // Helper functions, no changes needed
public:
    // Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            last_time = curr_time;
            curr_time = data_model->GetTime();

            std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
                      << ", Time: " << data_model->GetTime() << std::endl;

            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
                    send_steering_value = true;
                    send_throttle_value = false;
                    processImage( data_model->GetName(),
                                  data_model.get_typed<tronis::ImageSub>()->Image );
                    break;
                }
                case tronis::TronisDataType::ImageFrame:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFrameSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::ImageFramePose:
                {
                    send_steering_value = true;
                    send_throttle_value = false;
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::PoseVelocity:
                {
                    processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                    break;
                }
                case tronis::TronisDataType::BoxData:
                {
                    // cout << "Object detected !" << endl;
                    send_throttle_value = true;
                    send_steering_value = false;
                    /*is_object_insight =*/
                    processObject( data_model.get_typed<tronis::BoxDataSub>() );
                    break;
                }
                default:
                {
                    std::cout << data_model->ToString() << std::endl;
                    break;
                }
            }
            return true;
        }
        else
        {
            std::cout << data_model->ToString() << std::endl;
            return false;
        }
    }

protected:
    // Function to show an openCV image in a separate window
    void showImage( std::string image_name, cv::Mat& image )
    {
        cv::Mat out = image;  // shallow copy, why needed?
        if( image.type() == CV_32F || image.type() == CV_64F )
        {
            cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
        }
        cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );

        cv::imshow( image_name.c_str(), out );
        // cv::waitKey( 20 );
    }

    // Function to convert tronis image to openCV image
    bool processImage( const std::string& base_name, const tronis::Image& image )
    {
        std::cout << "processImage" << std::endl;
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_ = tronis::image2Mat( image );
        //// reduce the frequency of camera to reduce computational effort
        //// every 15 frames
        // static size_t cnt = 0;
        //// cout << "cnt is " << cnt << endl;
        // if( cnt == 15 )
        //{
        //    cnt = 0;
        //    return false;
        //}
        // else
        //{
        //    ++cnt;
        //}
        detectLanes();
#ifndef DEBUG_MOD
        showImage( image_name_, image_ );
#endif
        return true;
    }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    cv::setNumThreads( 8 );
    std::cout << "Welcome to lane assistant" << std::endl;

    // specify socket parameters
    std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
                  << "\", PortBind:" << socket_port << "}";

    int key_press = 0;  // close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param

    LaneAssistant lane_assistant;
	
    while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
        tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
            // wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
                // data received! reset timer
                time_ms = 0;

                // convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
                // identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
                // no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
