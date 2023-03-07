/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file imageMatcher.hpp
 * @brief Match two image using various method.
 * @version 1.0
 * @date 03-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __IMAGE_MATCHER__
#define __IMAGE_MATCHER__

#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "opencv4/opencv2/calib3d/calib3d.hpp"

#include "utility.hpp"

class imageMatcher
{
private:
/* Member Variables */
    bool m_b_initiated = false;
    // For Ratio test
    float m_f_ratio_th;
    // For SLIC-based Matching
    int    m_i_num_one_to_many; 
    int    m_i_num_keypoint_threshold;
    // Histogram voting for geopraphical constriant
    int    m_i_num_pixels_in_bin;
    cv::Point2i m_delta_translation;
    // Refinement matched result using template matching
    int    m_i_kdtree_flann_search_param;
    int    m_d_radius_of_pixel;   // 찾을 경계의 반경
    int    m_i_max_num_near_points;     // 찾을 주변점의 최대 개수
    int    m_i_boundary_gap;
    int    m_i_template_size;
    int    m_i_num_min_matched_pair;


public: /* Variables for Debuging */
    cv::Point2i m_p2i_trans;
    bool b_dbg_vis_close_key_point_extract = false;
public:
/* Init Functions */
    imageMatcher(/* args */);
    ~imageMatcher();
    bool init();
public:
/* Member Functions */
    bool MatchImages( cv::Mat in_img_1, cv::Mat in_img_2,
                      std::vector<cv::KeyPoint> in_keypoint_1, std::vector<cv::KeyPoint> in_keypoint_2,
                      cv::Mat in_desc_1, cv::Mat in_desc_2,
                      KeyDescType in_desc_type,
                      std::vector<cv::DMatch> &out_vec_dmatch);

    void ShowMatchingResult(cv::Mat in_img_1, cv::Mat in_img_2, 
                            std::vector<cv::KeyPoint> in_vec_keypoints_1, std::vector<cv::KeyPoint> in_vec_keypoints_2,
                            std::vector<cv::DMatch> in_vvec_dmatch, cv::Mat &out_img_matched);

    cv::Mat GetHomography( cv::Mat in_img_1, cv::Mat in_img_2,
                           std::vector<cv::KeyPoint> in_vec_keypoint_1, std::vector<cv::KeyPoint> in_vec_keypoint_2,
                           std::vector<cv::DMatch> in_vec_dmatch,
                           cv::Mat in_matched_img );


    bool GetCenterOfGeographyConstraint(std::vector<std::vector<cv::DMatch>> in_vvec_dmatch_reuslt,
                                        std::vector<cv::KeyPoint>            in_vec_uav_keypoints,
                                        std::vector<cv::KeyPoint>            in_vec_map_keypoints,
                                        cv::Point&                           out_center_location);

    bool RefineMatchedResult( cv::Mat in_uav_img, cv::Mat in_map_img, 
                              std::vector<cv::KeyPoint> in_vec_uav_keypoints, std::vector<cv::KeyPoint> in_vec_map_keypoints,
                              cv::Point in_translation,
                              std::vector<cv::DMatch> out_refined_matching_result);

    void ShowKeypoints(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_key_points, cv::Mat &out_mat_keypoint_img);

    bool SetGeographicConstraints(cv::Point2i in_p3d_uav_coordinate);

private:
/* Variables for Debugging */
    bool m_b_show_match_result;
    bool m_b_do_ratio_test;
    bool m_b_show_keypoint_result;
};

#endif //__IMAGE_MATCHER__