/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file ImageImport.hpp
 * @brief Image Import and get meta data from image
 * @version 1.0
 * @date 1-11-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __IMAGE_IMOPRT__
#define __IMAGE_IMOPRT__

#include "opencv4/opencv2/imgcodecs/imgcodecs.hpp"

class ImageImport
{
private:
    /* data */
public:
    ImageImport(/* args */);
    ~ImageImport();
    bool init();
};

ImageImport::ImageImport(/* args */)
{
}

ImageImport::~ImageImport()
{
}

bool ImageImport::init()
{

return true;
}


#endif