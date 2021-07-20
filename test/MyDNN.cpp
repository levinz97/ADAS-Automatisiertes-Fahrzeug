#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int smain()
{
    string classNames = "./Mydnn/coco.names";
    string weightPath = "./Mydnn/v3/frozen_inference_graph.pb";
    string configPath = "./Mydnn/v3/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
    //string configPath = "./Mydnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    //configPath = "./Mydnn/tflite_graph.pbtxt";
    //weightPath = "./Mydnn/tflite_graph.pb";
    dnn::Net net = dnn::readNetFromTensorflow( weightPath, configPath );
    //string model = "./Mydnn/mobilenetv2-7.onnx";
    //string model = "./Mydnn/mobilenet_v1_1.0_224.onnx";
    //dnn::Net net = dnn::readNetFromONNX( model );

	string fileName = "./Mydnn/data/curve01.png";
    Mat img = imread( fileName );
	imshow( "original picture", img );
    waitKey( 1 );

    Mat blob = dnn::blobFromImage( img, 1. / 127.5, Size( 300, 300 ), Scalar( 127.5, 127.5, 127.5 ),true );
    cout << "blob size " << blob.size << endl;

    net.setInput( blob );
    Mat output = net.forward();
    cout << "output size " << output.size << endl;
    Mat detectionMat( output.size[2], output.size[3], CV_32F, output.ptr<float>() );

    float confidenceThreshold = 0.5;
    for( int i = 0; i < detectionMat.rows; i++ )
    {
        float confidence = detectionMat.at<float>( i, 2 );
        cout << confidence << endl;
        if( confidence > confidenceThreshold )
        {
            size_t objectClass = ( size_t )( detectionMat.at<float>( i, 1 ) );

            int xLeftBottom = static_cast<int>( detectionMat.at<float>( i, 3 ) * img.cols );
            int yLeftBottom = static_cast<int>( detectionMat.at<float>( i, 4 ) * img.rows );
            int xRightTop = static_cast<int>( detectionMat.at<float>( i, 5 ) * img.cols );
            int yRightTop = static_cast<int>( detectionMat.at<float>( i, 6 ) * img.rows );

            ostringstream ss;
            ss << confidence;
            String conf( ss.str() );

            Rect object( (int)xLeftBottom, (int)yLeftBottom, (int)( xRightTop - xLeftBottom ),
                         (int)( yRightTop - yLeftBottom ) );

            rectangle( img, object, Scalar( 0, 255, 0 ), 2 );
            String label = /*String(classNames[objectClass]) +*/ char(objectClass) + ": " + conf;
            int baseLine = 0;
            Size labelSize = getTextSize( label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine );
            rectangle( img,
                       Rect( Point( xLeftBottom, yLeftBottom - labelSize.height ),
                             Size( labelSize.width, labelSize.height + baseLine ) ),
                       Scalar( 0, 255, 0 ), cv::FILLED );
            putText( img, label, Point( xLeftBottom, yLeftBottom ), FONT_HERSHEY_SIMPLEX, 0.5,
                     Scalar( 0, 0, 0 ) );
        }
    }
    imshow( "image", img );
    waitKey( 0 );
}