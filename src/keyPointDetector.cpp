/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file keyPointDetector.cpp
 * @brief Detect Keypoint of image using various method.
 * @version 1.0
 * @date 07-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#include "../include/keyPointDetector.hpp"

keyPointDetector::keyPointDetector()
{
    if( init() == false )
    {
        std::cout  << "[keyPointDetector] Class init has failed." << std::endl;
    }
    m_b_initiated = true;
}

keyPointDetector::~keyPointDetector(){}

bool keyPointDetector::init()
{

    // For SURF Features
    m_i_min_hessian        = 400;

    // For SLIC
    // SLIC Paramters
    m_i_num_superpixels    = 750;
    m_i_num_levels         = 4;
    m_i_prior              = 1;
    m_i_num_histogram_bins = 5;
    m_b_double_step        = false;

    m_i_num_iterations     = 4;
    m_i_keypoint_size      = 10;
    m_f_keypoint_angle     = 0.0;
    m_i_boundary_gap       = 10;

    return  true;
}

// Detect Key point from image following the input method
// Input  : Image for matching, key points type and descriptor type.
// Output : vector of keypoints and vector of Descriptor 
bool keyPointDetector::DetectKeyPoints(cv::Mat in_img, KeyDescType in_key_type, std::vector<cv::KeyPoint> &out_vec_keypoints)
{
    // Get duration for calculating
    std::clock_t start, end;

    if (in_img.empty())
    {
        std::cerr << "[keyPointDetector][DetectKeyPoints]Image is empty." << std::endl;
        return false;
    }

    // Get Keypoints from Image
    switch (in_key_type)
    {
        case TSIFT:
        {
            cv::Ptr<cv::SIFT> ptr_sift_detector = cv::SIFT::create();
            ptr_sift_detector->detect(in_img, out_vec_keypoints);
            break;
        }
        case TSURF:
        {
            cv::Ptr<cv::xfeatures2d::SURF> ptr_surf_detector = cv::xfeatures2d::SURF::create(m_i_min_hessian);
            ptr_surf_detector->detect(in_img, out_vec_keypoints);
            break;
        }
        case TBRIEF:
        {
            cv::Ptr<cv::xfeatures2d::StarDetector> ptr_star_detector = cv::xfeatures2d::StarDetector::create();
            ptr_star_detector->detect(in_img, out_vec_keypoints);
            break;
        }
        case TORB:
        {
            cv::Ptr<cv::ORB> ptr_orb_detector = cv::ORB::create();
            ptr_orb_detector->detect(in_img, out_vec_keypoints);
            break;
        }
        case TAKAZE:
        {
            cv::Ptr<cv::AKAZE> ptr_akaze_detector = cv::AKAZE::create();
            ptr_akaze_detector->detect(in_img, out_vec_keypoints);
            break;
        }
        case TSLIC:
        {
            GetSilcKeyPoints(in_img, false, out_vec_keypoints);
        }
        default:
        {
            std::cerr << "[keyPointDetector][DetectKeyPoint]The Keypoint Type is wrong." << std::endl;
            return false;
        }
    }
    if (out_vec_keypoints.empty())
    {
        std::cerr << "[keyPointDetector][DetectKeyPoints] The Keypoint vector is empty." << std::endl;
        return false;
    }

    if (m_b_show_keypoint_result)
    {
        cv::Mat mat_result_detected_keypoints;
        cv::drawKeypoints(in_img, out_vec_keypoints, mat_result_detected_keypoints);
        cv::imshow("Keypoint Result", mat_result_detected_keypoints);
        cv::waitKey();
        cv::destroyAllWindows();
    }
    
    // Time check end
    end = std::clock();
    double duration = (double)(end-start);
    std::cout << "Keypoint Detection duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;

    return true;
}

// Extract Key point  from image using SLIC boudnary.
// Input : Image to extract key point, downsample or not.
// Output: Keypoint Set vector.
bool keyPointDetector::GetSilcKeyPoints(cv::Mat in_img, bool do_downsample, std::vector<cv::KeyPoint> &out_vec_key_points )
{
    cv::Mat mat_slic_result, mat_slic_mask;
    cv::Mat mat_gray_img;
    cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
    int i_width, i_height, i_channels = in_img.channels();
    int i_display_mode = 0;
    i_width  = in_img.size().width;
    i_height = in_img.size().height;

    cv::cvtColor(in_img, mat_gray_img, cv::COLOR_BGR2GRAY);

    // Generate Initial SEEDs for SLIC.
    seeds = cv::ximgproc::createSuperpixelSEEDS( i_width, i_height, i_channels,
                                                 m_i_num_superpixels,
                                                 m_i_num_levels,
                                                 m_i_prior,
                                                 m_i_num_histogram_bins,
                                                 m_b_double_step);
    // Convert img type
    cv::Mat mat_converted_img;
    cv::cvtColor(in_img, mat_converted_img, cv::COLOR_BGR2HSV);

    // Calculates the superpixel sgmentation on a given image with parameters.
    seeds->iterate(mat_converted_img,m_i_num_iterations);
    mat_slic_result = in_img;

    // Get the segmented result.
    cv::Mat mat_labels;
    seeds->getLabels(mat_labels);

    // Get contour to use as key point.
    seeds->getLabelContourMask(mat_slic_mask,false);

    // Extaction Key points using SLIC Boundary mask and image
    std::vector<cv::KeyPoint> vec_key_points;
    ExtractKeypoints(mat_gray_img, mat_slic_mask, vec_key_points);
    out_vec_key_points = vec_key_points;

    std::cout << "Number of Key points from SLIC: " << vec_key_points.size() << std::endl;

    return true;

}

// Extract Key point from image using SLIC boudnary and SIFT.
// Input : Gray scale Img, slick mask.
// Output: Keypoint Set vector.
bool keyPointDetector::ExtractKeypoints(cv::Mat in_mat_gray_img, cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint>  &out_vector_keypoints)
{
    for(int idx_y = m_i_boundary_gap; idx_y < in_mat_slic_mask.rows-m_i_boundary_gap; idx_y++)
    {
        const unsigned char* Mi                      = in_mat_slic_mask.ptr<unsigned char>(idx_y                   );
        const unsigned char* ptr_gray_img_row_hight  = in_mat_gray_img .ptr<unsigned char>(idx_y - m_i_boundary_gap);
        const unsigned char* ptr_gray_img_row_mid    = in_mat_gray_img .ptr<unsigned char>(idx_y                   );
        const unsigned char* ptr_gray_img_row_bottom = in_mat_gray_img .ptr<unsigned char>(idx_y + m_i_boundary_gap);
        // TODO
        // Sampling method need to add if the algoritm runtime is too long.
        for(int idx_x = 2; idx_x < in_mat_slic_mask.cols-2; idx_x++)
        {
            int i_element      = Mi[idx_x];
            int i_left_pixel   = ptr_gray_img_row_mid   [idx_x - m_i_boundary_gap];
            int i_right_pixel  = ptr_gray_img_row_mid   [idx_x + m_i_boundary_gap];
            int i_upper_pixel  = ptr_gray_img_row_hight [idx_x                   ];
            int i_bottom_pixel = ptr_gray_img_row_bottom[idx_x                   ]; 
            if ( i_element      == 255  &&
                 i_left_pixel   != 0    &&
                 i_right_pixel  != 0    &&
                 i_upper_pixel  != 0    &&
                 i_bottom_pixel != 0     ) // SLIC Boundary output is 255 which is max of uchar. but why?
            {
                // Generate Key point object and add to vector.
                cv::Point    p_slick_point(idx_x, idx_y);
                cv::KeyPoint slic_key_point(p_slick_point, m_i_keypoint_size, m_f_keypoint_angle);
                out_vector_keypoints.push_back(slic_key_point);
            }
        }
    }
    if(out_vector_keypoints.empty() == true)
    {
        std::cerr << "Keypoint vector is empty!" << std::endl;
        return false;
    }
    return true;
}