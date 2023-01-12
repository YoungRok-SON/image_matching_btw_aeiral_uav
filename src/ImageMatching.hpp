/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file ImageMatching.hpp
 * @brief Match images btween UAV(low altitude) with Aerial Image Map(Hight Altitude)
 * @version 1.0
 * @date 1-11-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __IMAGE_MATCHING__
#define __IMAGE_MATCHING__

#include <iostream>
#include "opencv4/opencv2/core.hpp"

class ImageMatching
{
public:
    ImageMatching();
    ~ImageMatching();
public:
    bool RunImageMatching();
    bool SetImages();

private:
    /* Functions */
    bool Init();
    bool PreprocessUAVImg();
    bool KeyPointExtractionSLIC();
    bool DescriptorGenerationBRISK();
    bool MatchImages();
private:
    /* data */
    cv::Mat m_mat_uav_img;
    cv::Mat m_mat_map_img;
};





#endif