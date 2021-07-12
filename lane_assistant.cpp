#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>

#include "CurveFitting.hpp"
#include "PidController.hpp"
#define DEBUG_MODE
//#define DEBUG_STEERING
#define DEBUG_ACC
//#define DRAW_POLYGON

using namespace std;
using namespace cv;

class LaneAssistant
{
    // insert your custom functions and algorithms here
public:
    /*steering pid controller based on speed
        Kp,		Ki,		Kd
    -0.005, -1e-4, -0.01
    -0.003, -0.00005, -0.005 speed > 50 km/h
    -0.006, -0.00005, -0.001 low speed*/
    LaneAssistant()
        : steeringControll( -0.003, -0.00005, -0.005 ), throttleControll( -0.003, -0.00005, -0.005 )
    {
        left_last_fparam = 0;
        right_last_fparam = 0;
        width = height = 0;
        last_left_max = last_right_min = 0;
        center_of_lane = 360;
        leftlane_detected = rightlane_detected = true;
        curr_time = 0.;
        throttle_input = 0.;
        min_distance = numeric_limits<double>::max();
    }

    // do stuff with data
    // send results via socket
    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
        set_steering_input( socket );
        set_throttle_input( socket );
        return true;
    }
    void set_throttle_input( tronis::CircularMultiQueuedSocket& socket )
    {
        double err = min_distance - 5;
        throttleControll.UpdateErrorTerms( err );
        throttle_input = throttleControll.OutputToActuator( 1. );
        if( throttle_input > 1 )
            throttle_input = 1;
        if( throttle_input < 0 )
            throttle_input = 0;
        string prefix = "throttle,";
        socket.send( tronis::SocketData( prefix + to_string( throttle_input ) ) );

    }

	void set_steering_input(tronis::CircularMultiQueuedSocket& socket)
    {
		 // cout << "width_of_image = " << width << endl;
        double err = 0.;
        if( abs( width / 2. - center_of_lane ) > 1e-2 )
            err = width / 2. - center_of_lane;
        steeringControll.UpdateErrorTerms( err );
        double steering = steeringControll.OutputToActuator( 0.5 );
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
        //        steeringControll.setZero();
        //        zigzag_cnt = 0;
        //    }
        //}
        // last_direction_sign = steering > 0;
        if( !leftlane_detected && rightlane_detected )
        {
            steering = -0.3;
            steeringControll.setZero();
        }
        if( !rightlane_detected && leftlane_detected )
        {
            steering = 0.3;
            steeringControll.setZero();
        }

        // steering = 0.;

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
        //        // steeringControll.setZero();
        //        prefix = " right turn under steering,";
        //    }
        //    if( last_left_max < 0.3 * width && last_right_min < 0.55*width )
        //    {
        //        cout << "steering before is " << steering << endl;
        //        cout << "understeering detected, increase the steering!" << endl;
        //        steering = min(
        //            steering, ( last_left_max + last_right_min - width ) / ( 1. * width ) );
        //        // steeringControll.setZero();
        //        prefix = " left turn under steering,";
        //    }
        //}

        socket.send( tronis::SocketData( prefix + to_string( steering ) ) );
        // if( int( curr_time ) % 100 == 0 )
        //    steeringControll.setZero();
#ifdef DEBUG_STEERING
        cout << "center_of_lane = " << center_of_lane << endl;
        cout << "steering is " << steering << endl;
#endif
   
	}

protected:

    // lane detection
    std::string image_name_;
    cv::Mat image_, grey_image;
    int width, height;  // size of under half of image
    Vec3f left_last_fparam, right_last_fparam;

    // lane keeping
    double center_of_lane;
    double steering;
    PidController steeringControll;
    bool leftlane_detected, rightlane_detected;
    double curr_time;

    // adaptive cruise controll
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;
    double throttle_input;
    double min_distance;
    vector<double> all_distance;
    PidController throttleControll;

    int last_left_max, last_right_min;  // used for plot a more stable lane

    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        cvtColor( image_, grey_image, COLOR_BGR2GRAY );
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

        // remove the line on the hood
        Mat region_on_hood = Mat::zeros( height, width, grey_image.type() );
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
            line( res_Hough, pt1, pt2, Scalar( 0, 0, 255 ), 3, LINE_AA );
            double slope = ( 1. * l[3] - l[1] ) / ( l[2] - l[0] );
            ////-------------------------------------------------------------------------------------important_parameters------------------------------------///
            if( abs( slope ) < tan( CV_PI / 180 * 10 ) )
                continue;
            left_min = min( left_min, pt1.x );
            right_max = max( right_max, pt2.x );
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
        CurveFitting* fit_L_ptr = generate_oneLine( left_lines, "left" );
        CurveFitting* fit_R_ptr = generate_oneLine( right_lines, "right" );
        leftlane_detected = ( fit_L_ptr != nullptr );
        rightlane_detected = ( fit_R_ptr != nullptr );

#ifdef DRAW_POLYGON
        draw_polygon( fit_L_ptr, fit_R_ptr );
#endif
        double left_point = findLinePoint( fit_L_ptr, "left" );
        double right_point = findLinePoint( fit_R_ptr, "right" );
        // cout << "left_point " << left_point << "right_point " << right_point << endl;
        if( isfinite( left_point ) && isfinite( right_point ) )
            center_of_lane = ( left_point + right_point ) / 2.;
        if( center_of_lane < 0 )
            center_of_lane = 0;
        if( center_of_lane > width )
            center_of_lane = width;
        line( image_, Point2f( center_of_lane, 1.7 * height ), Point2f( width / 2, 2 * height ),
              Scalar( 104, 55, 255 ), 3 );

        delete fit_L_ptr;
        delete fit_R_ptr;
    }
    /* find the intersect point between lane and bottom line, used to compute the center of lane.
     * i.e., find x for y = ax^2 + bx + c where y == height of total image_*/
    double findLinePoint( const CurveFitting* fit_ptr, string TypeOfLane )
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
            if( TypeOfLane == "right" && lane_point < 0 )
                lane_point = temp_x_2 / ( 2 * a );
        }
        // cout << "lane point = " << lane_point << endl;
        return lane_point;
    }
    CurveFitting* generate_oneLine( vector<Point2f>& lines, string typeOfLines )
    {
        if( lines.empty() )
        {
            cout << "no " << typeOfLines << " line detected!" << endl;
            return nullptr;
        }
#ifdef DEBUG_MODE
        Scalar color = Scalar( 255, 20, 20 );
        color = typeOfLines == "right" ? Scalar( 100, 100, 255 ) : Scalar( 255, 100, 0 );
        for( auto point : lines )
        {
            circle( image_, point, 10, color, 3 );
        }
        // imshow( typeOfLines + " points", image_ );
        // waitKey();
#endif
        CurveFitting* fit_ptr;
        if( typeOfLines == "left" )
        {
            fit_ptr = new CurveFitting( lines, left_last_fparam );
            fit_ptr->solve( 30 );
            left_last_fparam = fit_ptr->param;
            // cout << "last left fitting parameter is " << left_last_fparam << endl;
            draw_polynomial( fit_ptr, typeOfLines );
        }
        else
        {
            fit_ptr = new CurveFitting( lines, right_last_fparam );
            fit_ptr->solve( 30 );
            right_last_fparam = fit_ptr->param;
            // cout << "last right fitting parameter is " << right_last_fparam << endl;
            draw_polynomial( fit_ptr, typeOfLines );
        }
        return fit_ptr;
    }

    void draw_polynomial( const CurveFitting* fit, string typeOfLines )
    {
        if( !fit )
            return;
        // Scalar color = Scalar( 104, 55, 255, 10 );
        Scalar color = Scalar( 255, 20, 20 );
#ifdef DEBUG_MODE
        color = typeOfLines == "right" ? Scalar( 100, 100, 255 ) : Scalar( 255, 100, 0 );
        //        for( auto item : lines )
        //        {
        //            cout << "points is : " << item << endl;
        //        }
#endif
        float start_point_x, end_point_x;
        if( typeOfLines == "left" )
        {
            start_point_x = 0;
            end_point_x = last_left_max + 0.1 * ( fit->max_x - last_left_max );

#ifdef DEBUG_MODE
            end_point_x = fit->max_x;
#endif
            last_left_max = end_point_x;
        }
        else
        {
            start_point_x = last_right_min + 0.1 * ( fit->min_x - last_right_min );

#ifdef DEBUF_MODE
            start_point_x = fit->min_x;
#endif

            last_right_min = start_point_x;
            end_point_x = width;
        }
        vector<Point2f> curvePoints;
        for( float x = start_point_x; x <= end_point_x; x += 5 )
        {
            float y = fit->computeValue( x );
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

    // draw the areas that have been enclosed by 2 lanes, only used for visualisation.
    void draw_polygon( const CurveFitting* fitL, const CurveFitting* fitR )
    {
        if( !fitL || !fitR )
        {
            cout << "no fit ptr" << endl;
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
        ego_velocity_ = msg->Velocity * 3.6 * 1e-2; // in Km/h 
#ifdef DEBUG_ACC
        //cout << "ego_location is " << ego_location_.ToString() << " \n ego_orientation is "
        //     << ego_orientation_.ToString() << "\n ego_velocity is " << ego_velocity_ << endl;
#endif
        return true;
    }

    bool processObject( tronis::BoxDataSub* sensorData )
    {
        // process data from ObjectListsSensor
        for( size_t i = 0; i < sensorData->Objects.size(); i++ )
        {
            const tronis::ObjectSub& object = sensorData->Objects[i];
#ifdef DEBUG_ACC
            cout << object.ActorName.Value() << " at ";
            cout << object.Pose.Location.ToString() << endl;
#endif
            tronis::LocationSub location = object.Pose.Location;
            tronis::QuaternionSub orientation = object.Pose.Quaternion;
            float pos_x = location.X / 100;
            float pos_y = location.Y / 100;
            double dist = sqrt( pow( pos_x, 2 ) + pow( pos_y, 2 ) );
            float angle = atan( pos_y / pos_x );
            min_distance = min( dist, min_distance );
            if( object.Type )
            {
            }
            else
            {
                if( dist < 2 && angle < CV_PI / 4 )
                    throttle_input = 0.;
            }
        }

        return true;
    }

    // Helper functions, no changes needed
public:
    // Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            curr_time = data_model->GetTime();

            std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
                      << ", Time: " << data_model->GetTime() << std::endl;

            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
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
                case tronis::TronisDataType::Object:
                {
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
    void showImage( std::string image_name, cv::Mat image )
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
//
// int main()
//{
//    test();
//}
