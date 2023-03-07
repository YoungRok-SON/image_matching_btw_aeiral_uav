/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file keyPointDetector.hpp
 * @brief Detect Keypoint of image using various method.
 * @version 1.0
 * @date 07-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __KEY_POINT_DETECTOR__
#define __KEY_POINT_DETECTOR__

#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"
#include "opencv4/opencv2/xfeatures2d.hpp"
#include "opencv4/opencv2/ximgproc.hpp"

#include "utility.hpp"



class keyPointDetector
{
private:
/* Member Variables */
    // Class init
    bool m_b_initiated = false;

    // For SURF Features
    int m_i_min_hessian;

    // For SLIC
    // SLIC Paramters
    int   m_i_num_superpixels;
    int   m_i_num_levels;
    int   m_i_prior;
    int   m_i_num_histogram_bins;
    bool  m_b_double_step;
    int   m_i_num_iterations;
    int   m_i_keypoint_size;
    int   m_f_keypoint_angle;
    int   m_i_boundary_gap;

public:
/* Init Functions */
    keyPointDetector(/* args */);
    ~keyPointDetector();
public:
/* Member Functions */
    bool DetectKeyPoints (cv::Mat in_img, KeyDescType in_key_type, std::vector<cv::KeyPoint> &out_vec_keypoints);
    bool init();
    bool GetSilcKeyPoints(cv::Mat in_img, bool do_downsample, std::vector<cv::KeyPoint> &out_vec_key_points );
    bool ExtractKeypoints(cv::Mat in_mat_gray_img, cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint>  &out_vector_keypoints);

private:
/* Variables for Debugging */
    bool m_b_show_keypoint_result = false;
};


#endif //__KEY_POINT_DETECTOR__