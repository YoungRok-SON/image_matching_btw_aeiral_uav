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

#include "opencv2/core.hpp" // Mat, keypoint, etc ... class
#include "opencv2/ximgproc.hpp" // for SLIC algorithm
#include "opencv2/highgui/highgui.hpp" // This header will be removed after debuging.

#include <time.h>

enum ImgType
{
    UAV,
    MAP
};

class ImageMatching
{
public:
    ImageMatching();
    ~ImageMatching();

public: /* Main Funtions */
    bool RunImageMatching();
    bool SetImages();

public: /* Functions */ 
    bool Init();
    bool ComputeKeyDescriptorSlicBrisk(cv::Mat in_img, ImgType in_img_type, bool do_downsample,
                                        std::vector<cv::KeyPoint> &out_vec_key_points, cv::Mat &out_mat_descriptors);
    bool MatchImages(std::vector<cv::KeyPoint> in_vec_uav_keypoint, std::vector<cv::KeyPoint> in_vec_map_keypoint,
                                cv::Mat in_mat_uav_descriptors, cv::Mat in_mat_map_descriptors);
    bool GetCenterOfGeographyConstraint(std::vector<std::vector<cv::DMatch>> in_vvec_dmatch_reuslt,
                                        std::vector<cv::KeyPoint>            in_vec_uav_keypoints,
                                        std::vector<cv::KeyPoint>            in_vec_map_keypoints,
                                        cv::Point2f &out_center_location                            );
    void ShowKeypoints(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_key_points, cv::Mat &out_mat_keypoint_img);
private:
    bool ExtractKeypoints(cv::Mat in_mat_gray_img, cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint> &out_vector_keypoints);
private: /* Variables */
    bool    m_b_initiated = false;
    cv::Mat m_mat_uav_img;
    cv::Mat m_mat_map_img;
    // SLIC Paramters
    int   m_i_uav_num_iterations     = 4;
    int   m_i_uav_prior              = 2;
    bool  m_b_uav_double_step        = false;
    int   m_i_uav_num_superpixels    = 400;
    int   m_i_uav_num_levels         = 4;
    int   m_i_uav_num_histogram_bins = 10;
    // Key point extraction parameters
    int   m_i_threshold     = 30;
    int   m_i_octaves       = 4;
    float m_f_pattern_scale = 1.0f;
    int   m_i_boundary_gap  = 5;
    int   m_i_keypoint_size = 1;
    // Descriptor matching parameters
    int   m_i_num_one_to_many        = 100; 
    int   m_i_num_keypoint_threshold = 1000;
    // Histogram voting for geopraphical constriant
    // int   m_i_hist_size = histogram size is depends on map image size.
    int   m_i_num_pixels_in_bin      = 3;
};





#endif