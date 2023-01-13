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
    bool DescriptorGenerationBRISK();
    bool MatchImages();

private:
    bool ExtractKeypoints(cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint> &out_vector_keypoints);

private: /* Variables */
    bool    m_b_initiated = false;
    cv::Mat m_mat_uav_img;
    cv::Mat m_mat_map_img;

    int   m_i_uav_num_iterations = 4;
    int   m_i_uav_prior = 2;
    bool  m_b_uav_double_step = false;
    int   m_i_uav_num_superpixels = 400;
    int   m_i_uav_num_levels = 4;
    int   m_i_uav_num_histogram_bins = 5;
    
    int   m_i_threshold     = 30;
    int   m_i_octaves       = 4;
    float m_f_pattern_scale = 1.0f;
};





#endif