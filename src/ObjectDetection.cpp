#include <chrono>
#include <opencv2/opencv.hpp>
#include "ObjectDetection.hpp"
//#define DEBUG

using namespace cv;
using namespace std;

bool ObjectDetection::detectObject( Mat& input_image, vector<Rect>& object_lists, int hight_offset )
{
    /*if( countNonZero( input_image ) < 1 )
        return false;*/
    // if detectObject is called clear the stored old objects
    // because detectObject is not called every frame to reduce the computational effort
    detected_objects.clear();
    object_lists.clear();

    if( input_image.channels() == 4 )
    {
        // if RGBA, convert to RGB
		// in tronis the camera type must be set to RGBA
        cvtColor( input_image, input_image, cv::COLOR_RGBA2RGB );
    }
#ifdef DEBUG
    auto t1 = chrono::high_resolution_clock::now();
#endif
    Mat blob = dnn::blobFromImage( input_image, 1. / 127.5, Size( 300, 300 ),
                                   Scalar( 127.5, 127.5, 127.5 ), true );
    net.setInput( blob );
    Mat output = net.forward();
#ifdef DEBUG
    auto t2 = chrono::high_resolution_clock::now();
    cout << "blob size is " << blob.size << endl;
    cout << "output size " << output.size << endl;
    auto duration = chrono::duration_cast<chrono::milliseconds>( t2 - t1 );
    cout << "duration is " << duration.count() << " ms" << endl;
#endif
    Mat detectionMat( output.size[2], output.size[3], CV_32F, output.ptr<float>() );

    for( size_t i = 0; i < detectionMat.rows; ++i )
    {
        float confidence = detectionMat.at<float>( i, 2 );
        if( confidence >= confidence_threshold )
        {
            int object_class = (int)detectionMat.at<float>( i, 1 );
            string className = myClassNames::classNames[object_class - 1];
#ifdef DEBUG
            cout << "object class is " << className << " prob = " << to_string( confidence )
                 << endl;
#endif
            if( className == "boat" || className == "bus" )
                object_class = 3;
            int xLeftBottom = static_cast<int>( detectionMat.at<float>( i, 3 ) * input_image.cols );
            int yLeftBottom = static_cast<int>( detectionMat.at<float>( i, 4 ) * input_image.rows +
                                                hight_offset );
            int xRightTop = static_cast<int>( detectionMat.at<float>( i, 5 ) * input_image.cols );
            int yRightTop = static_cast<int>( detectionMat.at<float>( i, 6 ) * input_image.rows +
                                              hight_offset );

            Rect object( xLeftBottom, yLeftBottom, xRightTop - xLeftBottom,
                         yRightTop - yLeftBottom );
            if( object.area() > hight_offset * 0.15 * 360 )
                continue;
            object_lists.push_back( object );
            detected_objects.push_back( {object_class, confidence} );
        }
    }
    //	imshow( "image", input_image );
    //  waitKey( );
    // cvtColor( input_image, input_image, cv::COLOR_RGB2RGBA );
    if( object_lists.empty() )
        return false;

    return true;
}

void ObjectDetection::drawBoundingBox( Mat& input_image, vector<Rect>& object_lists )
{
    int i = 0;
    for( const auto& object : object_lists )
    {
        int object_class = detected_objects[i].first;
        string conf = to_string( int( detected_objects[i].second * 100 ) );
        string className = myClassNames::classNames[object_class - 1];
        rectangle( input_image, object, Scalar( 0, 255, 0 ), 2 );
        // label used to display
        string label = className + ":" + conf + "%";
        int baseLine = 0;
        Size labelSize = getTextSize( label, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1, &baseLine );

        rectangle( input_image,
                   Rect( Point( object.x, object.y - labelSize.height ),
                         Size( labelSize.width, labelSize.height + baseLine ) ),
                   Scalar( 0, 255, 0 ), cv::FILLED );
        putText( input_image, label, Point( object.x, object.y ), FONT_HERSHEY_SIMPLEX, 0.5,
                 Scalar( 0, 0, 0 ) );
        ++i;
    }
}
