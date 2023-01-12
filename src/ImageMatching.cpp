/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file ImageMatching.cpp
 * @brief Match images btween UAV(low altitude) with Aerial Image Map(Hight Altitude)
 * @version 1.0
 * @date 1-11-2023
 * @bug No known bugs
 * @warning No warnings
 */

#include "ImageMatching.hpp"

ImageMatching::ImageMatching(/* args */)
{
    if( Init() == true )
        std::cout  << "Image Matching Class Init has failed." << std::endl;
}

ImageMatching::~ImageMatching() {}

bool ImageMatching::Init()
{
    return true;
}

bool ImageMatching::RunImageMatching();


bool ImageMatching::PreprocessUAVImg()
{

}
bool ImageMatching::KeyPointExtractionSLIC()
{

}
bool ImageMatching::DescriptorGenerationBRISK()
{

}
bool MatchImages()
{

}