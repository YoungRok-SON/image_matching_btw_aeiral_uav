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

#include "opencv4/opencv2/core.hpp" // Mat, keypoint, etc ... class
#include "opencv4/opencv2/ximgproc.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp" // This header will be removed after debuging.
#include "opencv4/opencv2/imgproc/imgproc.hpp" // for cvtcolor function
// #include "opencv4/opencv2/xfeatures2d.hpp" // for SIFT Feature
#include "opencv4/opencv2/features2d/features2d.hpp"

#include <time.h>

enum ImgType
{
    UAV,
    MAP
};

enum DescriptorType
{
    SIFT,
    BRISK
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
    bool ComputeKeyDescriptorSlic(cv::Mat in_img, ImgType in_img_type, bool do_downsample, DescriptorType in_descriptor_type,
                                  std::vector<cv::KeyPoint> &out_vec_key_points, cv::Mat &out_mat_descriptors);

    bool MatchImages(cv::Mat in_uav_img, cv::Mat in_map_img, 
                     std::vector<cv::KeyPoint> in_vec_uav_keypoint, std::vector<cv::KeyPoint> in_vec_map_keypoint,
                     cv::Mat in_mat_uav_descriptors, cv::Mat in_mat_map_descriptor, DescriptorType in_descriptor_type);

    bool GetCenterOfGeographyConstraint(std::vector<std::vector<cv::DMatch>> in_vvec_dmatch_reuslt,
                                        std::vector<cv::KeyPoint>            in_vec_uav_keypoints,
                                        std::vector<cv::KeyPoint>            in_vec_map_keypoints,
                                        cv::Point &out_center_location                            );

    bool RefineMatchedResult( cv::Mat in_uav_img, cv::Mat in_map_img, 
                              std::vector<cv::KeyPoint> in_vec_uav_keypoints, std::vector<cv::KeyPoint> in_vec_map_keypoints,
                              cv::Point in_translation,
                              std::vector<cv::DMatch> out_refined_matching_result);

    void ShowKeypoints(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_key_points, cv::Mat &out_mat_keypoint_img);
private:
    bool ExtractKeypoints(cv::Mat in_mat_gray_img, cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint> &out_vector_keypoints);
private: /* Variables */
    // Init
    bool    m_b_initiated = false;
    cv::Mat m_mat_uav_img;
    cv::Mat m_mat_map_img;
    // SLIC Paramters
    int   m_i_uav_num_superpixels    = 750;
    int   m_i_uav_num_levels         = 4;
    int   m_i_uav_prior              = 1;
    int   m_i_uav_num_histogram_bins = 5;
    bool  m_b_uav_double_step        = false;
    int   m_i_uav_num_iterations     = 4;

    int   m_i_map_num_superpixels    = 3000;
    int   m_i_map_num_levels         = 3;
    int   m_i_map_prior              = 1;
    int   m_i_map_num_histogram_bins = 10;
    bool  m_b_map_double_step        = false;
    int   m_i_map_num_iterations     = 10;
    // Key point extraction parameters
    int   m_i_keypoint_size  = 10;
    int   m_f_keypoint_angle = 0.0;
    /// for brisk
    int   m_i_threshold      = 30;
    int   m_i_octaves        = 4;
    float m_f_pattern_scale  = 1.0f;
    int   m_i_boundary_gap   = 5;
    /// for sift
    int    m_i_n_feature      = 0;
    int    m_i_n_octav_layers = 3;
    double m_d_contrast_th    = 0.04;
    double m_d_edge_th        = 10;
    double m_d_sigma          = 1.6;
    // Descriptor matching parameters

    int   m_i_num_one_to_many        = 500; 
    int   m_i_num_keypoint_threshold = 1000;
    // Histogram voting for geopraphical constriant
    int   m_i_num_pixels_in_bin      = 1;



public: /* Variables for Debuging */
    cv::Point2i m_p2i_trans;
    bool b_dbg_vis_close_key_point_extract = true;
    
};

#endif