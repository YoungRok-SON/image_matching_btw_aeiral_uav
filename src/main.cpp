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

// Debug Variables
bool   m_b_visualize   = true;

int main()
{
	// Get Image and Preprocess using UAV States.
	ImagePreprocessing IP;
	cv::Mat m_mat_uav_img, m_mat_map_img;
	std::vector<int> m_veci_target_resize_size;

	// Get Preprocessing Image.
	IP.GetImages(m_mat_uav_img, m_mat_map_img);
	if (m_mat_uav_img.empty() || m_mat_map_img.empty()) {
		std::cerr << "Image load failed!" << std::endl;
		return -1;
	}

	// Vizualization for image import
	if (m_b_visualize == true)
	{
		cv::imshow ("Map Image", m_mat_map_img);
		cv::imshow ("UAV Image", m_mat_uav_img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	// Get center of submap
	cv::Point2i p2i_img_size {m_mat_map_img.rows, m_mat_map_img.cols};
	cv::Point2i p2i_submap_center {0, 0};
	IP.GetCenterOfSubMap(IP.m_p2d_uav_lonlat_relative, IP.m_p2d_range_lonlat, p2i_img_size, p2i_submap_center);
	if (m_b_visualize == true)
	{
		cv::Mat mat_center_of_submap;
		m_mat_map_img.copyTo(mat_center_of_submap);
		cv::circle(mat_center_of_submap, p2i_submap_center, 5, cv::Scalar(0,0,255), -1);
		cv::imwrite("Center of Submap.png", mat_center_of_submap);
		cv::imshow ("Submap Center Image", mat_center_of_submap);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	// Get Submap using center point and altitude of uav.
	cv::Mat mat_submap;
	IP.GetSubMap(m_mat_map_img, p2i_submap_center, IP.m_f_altitude_uav, mat_submap);
	if (m_b_visualize == true)
	{

		cv::imwrite("Submap of the map.png", mat_submap);
		cv::imshow ("Submap  Image", mat_submap);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	// Image Matching Class Initiation.
	ImageMatching IM;
	
	// Extract UAV image key point and descriptor using SLIC and BRISK.
	std::vector<cv::KeyPoint> vec_uav_key_points;
	cv::Mat mat_uav_descriptors;
	bool b_do_downsample_uav = true;
	IM.ComputeKeyDescriptorSlic(m_mat_uav_img, UAV, b_do_downsample_uav, SIFT, vec_uav_key_points, mat_uav_descriptors);

	// Extact MAP image key point and descriptor using SLIC and BRISK.
	std::vector<cv::KeyPoint> vec_map_key_points;
	cv::Mat mat_map_descriptors;
	bool b_do_downsample_map = false;
	IM.ComputeKeyDescriptorSlic(mat_submap, MAP, b_do_downsample_map, SIFT, vec_map_key_points, mat_map_descriptors);

	
	cv::Mat mat_uav_key_points_img;
	cv::Mat mat_map_key_points_img;
	// Vizualization for SLIC result
	if (m_b_visualize == true)
	{
		// UAV Key point result visualization.
		IM.ShowKeypoints(m_mat_uav_img, vec_uav_key_points, mat_uav_key_points_img);
		// Map Key point result visualization.
		IM.ShowKeypoints(mat_submap, vec_map_key_points,mat_map_key_points_img);
		cv::imshow("MAP Keypoint Image", mat_map_key_points_img);
		cv::imshow("UAV Keypoint Image", mat_uav_key_points_img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}	

	// Image matching using two keypoints and descriptors.
	IM.MatchImages(m_mat_uav_img, mat_submap, vec_uav_key_points, vec_map_key_points, mat_uav_descriptors, mat_map_descriptors, SIFT);

	// Visualization for histogram voting verification.
	if (m_b_visualize == true)
	{
		cv::Mat mat_map_histogram_verification_img;
		int i_uav_width, i_uav_height, i_uav_center_x, i_uav_center_y, i_moved_center_x, i_moved_center_y;
		i_uav_width      = m_mat_uav_img.cols;
		i_uav_height     = m_mat_uav_img.rows;
		i_uav_center_x   = i_uav_width/2;
		i_uav_center_y   = i_uav_height/2;
		i_moved_center_x = i_uav_center_x - IM.m_p2i_trans.x;
		i_moved_center_y = i_uav_center_y - IM.m_p2i_trans.y;

		mat_map_key_points_img.copyTo(mat_map_histogram_verification_img);
		cv::circle(mat_map_histogram_verification_img, cv::Point(i_moved_center_x, i_moved_center_y), 5, cv::Scalar(0,0,255), -1);
		cv::rectangle(mat_map_histogram_verification_img, cv::Rect(-IM.m_p2i_trans.x, -IM.m_p2i_trans.y, i_uav_width, i_uav_height), cv::Scalar(0,0,255), 1, 8, 0);
		cv::imwrite("histogram voting constraint verification.png", mat_map_histogram_verification_img);
		cv::imshow("MAP Keypoint Image", mat_map_histogram_verification_img);
		cv::imshow("UAV Image", m_mat_uav_img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}	



	return 0;
}