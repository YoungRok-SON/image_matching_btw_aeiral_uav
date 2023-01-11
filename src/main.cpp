/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file main.cpp
 * @brief import two image to match and match.
 * @version 1.0
 * @date 1-11-2023
 * @bug No known bugs
 * @warning No warnings
 */

#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"

int main()
{
    // About UAV iamge loading
    std::string uav_img_path         = "../01_uav_images/orthophotos_100m/";
    std::string uav_img_file_name    = "DJI_0378.JPG";
    std::string uav_img_path_name    = uav_img_path + uav_img_file_name ;
	cv::Mat img = cv::imread(uav_img_path_name);

	if (img.empty()) {
		std::cerr << "Image laod failed!" << std::endl;
		return -1;
	}

	cv::namedWindow("image");
	cv::imshow("image", img);
	cv::waitKey();
	cv::destroyAllWindows();
	
	return 0;
}