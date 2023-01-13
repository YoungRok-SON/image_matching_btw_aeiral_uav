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

#include "opencv2/highgui/highgui.hpp"

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
	cv::waitKey(0);
	cv::destroyAllWindows();

<<<<<<< HEAD
	// Image Matching Class Initiation.
	ImageMatching IM;
	
	// Extract key point and descriptor using SLIC and BRISK.
	std::vector<cv::KeyPoint> vec_uav_key_points;
	cv::Mat mat_uav_descriptors;
	bool b_do_downsample = true;
	IM.ComputeKeyDescriptorSlicBrisk(m_mat_uav_img, UAV,  b_do_downsample, vec_uav_key_points, mat_uav_descriptors);
=======
	// Match two images
	ImageMatching IM;
>>>>>>> 3269001ad0ced71cb36485629e9baea36357200a
	
	return 0;
}