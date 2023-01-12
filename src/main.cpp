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

#include "ImagePreprocessing.hpp"
#include "ImageMatching.hpp"

#include "opencv4/opencv2/highgui/highgui.hpp"

int main()
{
	// Get Image and Preprocess using UAV States.
	ImagePreprocessing II;
	cv::Mat m_mat_uav_img, m_mat_map_img;
	std::vector<int> m_veci_target_resize_size;

	// Get Preprocessing Image.
	II.GetImages(m_mat_uav_img, m_mat_map_img);
	if (m_mat_uav_img.empty() || m_mat_map_img.empty()) {
		std::cerr << "Image load failed!" << std::endl;
		return -1;
	}
	cv::imshow("Map Image", m_mat_map_img);
	cv::imshow("UAV Image", m_mat_uav_img);
	cv::waitKey();
	cv::destroyAllWindows();

	// Match two images
	ImageMatching IM;
	
	return 0;
}