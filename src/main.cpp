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

	// Image Matching Class Initiation.
	ImageMatching IM;
	
	// Extract UAV image key point and descriptor using SLIC and BRISK.
	std::vector<cv::KeyPoint> vec_uav_key_points;
	cv::Mat mat_uav_descriptors;
	bool b_do_downsample_uav = true;
	IM.ComputeKeyDescriptorSlicBrisk(m_mat_uav_img, UAV,  b_do_downsample_uav, vec_uav_key_points, mat_uav_descriptors);

	// Extact MAP image key point and descriptor using SLIC and BRISK.
	std::vector<cv::KeyPoint> vec_map_key_points;
	cv::Mat mat_map_descriptors;
	bool b_do_downsample_map = false;
	IM.ComputeKeyDescriptorSlicBrisk(m_mat_map_img, UAV,  b_do_downsample_map, vec_map_key_points, mat_map_descriptors);

	// UAV Key point result visualization.
	cv::Mat mat_uav_key_points_img;
	IM.ShowKeypoints(m_mat_uav_img, vec_uav_key_points, mat_uav_key_points_img);
	// Map Key point result visualization.
	cv::Mat mat_map_key_points_img;
	IM.ShowKeypoints(m_mat_map_img, vec_map_key_points,mat_map_key_points_img);

	// Show Image
	cv::imshow("MAP Keypoint Image", mat_map_key_points_img);
	cv::imshow("UAV Keypoint Image", mat_uav_key_points_img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Image matching using two keypoints and descriptors.
	IM.MatchImages(vec_uav_key_points, vec_map_key_points, mat_uav_descriptors, mat_map_descriptors);




	return 0;
}