#pragma once
#include <iostream>
#include <vector>
#include <opencv2/dnn.hpp>
#include "classNames.hpp"

class ObjectDetection
{
private:
    std::string weightPath;
    std::string configPath;
    cv::dnn::Net net;
    float confidence_threshold;
    std::vector<std::pair<int, float>> detected_objects;

public:
    /**
	 * constructor
	 * @param weight path .bp file, must be a frozen_inference_graph
	 * @param configuration path .pbtxt file
	 * @param the confidence threshold for object detection, default threshold = 50%
     */
    ObjectDetection( std::string& weightPath, std::string& configPath, float threshold = 0.5 )
        : confidence_threshold( threshold )
    {
        this->weightPath = weightPath;
        this->configPath = configPath;
        net = cv::dnn::readNetFromTensorflow( weightPath, configPath );
        if( net.empty() )
        {
            std::cout << "input not valid" << std::endl;
        }
        detected_objects = {};
    }
    /**
     * detect the object in camera
     * @param input image, used to detect
     * @param object lists, used to store the result
     * @param offset, when only part of original picture used for detection
     * @output detect if there are objects insight
     */
    bool detectObject( cv::Mat& input_image, std::vector<cv::Rect>& object_lists,
                       int hight_offset );

    /**
     * draw the boudingbox of detected object on the image
     * @param input image, used to draw boundingbox
     * @param object lists, used to read the stored boundingbox(cv::Rect)
     */
    void drawBoundingBox( cv::Mat& input_image, std::vector<cv::Rect>& objectLists );
};
