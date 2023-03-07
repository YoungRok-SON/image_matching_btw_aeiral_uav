
/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file descriptorComputer.cpp
 * @brief Compute descriptor of keypoints using various method.
 * @version 1.0
 * @date 07-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#include "../include/descriptorComputer.hpp"

descriptorComputer::descriptorComputer()
{

}

descriptorComputer::~descriptorComputer(){}

bool descriptorComputer::init()
{
    // Key point extraction parameters
    m_i_keypoint_size  = 10;
    m_f_keypoint_angle = 0.0;
    // for slic-based SIFT Keypoints 
    m_i_n_feature      = 0;
    m_i_n_octav_layers = 3;
    m_d_contrast_th    = 0.04;
    m_d_edge_th        = 10;
    m_d_sigma          = 1.6;
    
    return true;
}

// Compute Descriptor of keypoints using various method.
// Input  : Image, key points and  and type of descriptor.
// Output : descriptor matrix form of cv::Mat.
bool descriptorComputer::ComputeDescriptors(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_keypoints, KeyDescType in_desc_type, cv::Mat &out_mat_descriptors)
{
    if (in_img.empty())
    {
        std::cerr << "[descriptorComputer][ComputeDescriptors] Image is empty." << std::endl;
        return false;
    }
    // Get duration for calculating
    std::clock_t start, end;

    // Get Descriptor from Key points
    switch (in_desc_type)
    {
        case TSIFT:
        {
            cv::Ptr<cv::SIFT> ptr_sift_detector = cv::SIFT::create();
            ptr_sift_detector->compute(in_img, in_vec_keypoints, out_mat_descriptors);
            
            break;
        }
        case TSURF:
        {
            cv::Ptr<cv::xfeatures2d::SURF> ptr_surf_detector = cv::xfeatures2d::SURF::create();
            ptr_surf_detector->compute(in_img, in_vec_keypoints, out_mat_descriptors);
            break;
        }
        case TBRIEF:
        {
            cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> ptr_brief_detector = cv::xfeatures2d::BriefDescriptorExtractor::create();
            ptr_brief_detector->compute(in_img, in_vec_keypoints, out_mat_descriptors);
            break;
        }
        case TORB:
        {
            cv::Ptr<cv::ORB> ptr_orb_detector = cv::ORB::create();
            ptr_orb_detector->compute(in_img, in_vec_keypoints, out_mat_descriptors);
            break;
        }
        case TAKAZE:
        {
            cv::Ptr<cv::AKAZE> ptr_orb_detector = cv::AKAZE::create();
            ptr_orb_detector->compute(in_img, in_vec_keypoints, out_mat_descriptors);
            break;
        }
        case TSLIC:
        {
            cv::Ptr<cv::SIFT> ptr_sift_feature = cv::SIFT::create(m_i_n_feature, m_i_n_octav_layers, m_d_contrast_th, m_d_edge_th, m_d_sigma);
            ptr_sift_feature->compute(in_img, in_vec_keypoints, out_mat_descriptors);
        }
        default:
        {
            std::cerr << "[descriptorComputer][ComputeDescriptors]]The Keypoint Type is wrong." << std::endl;
            return false;
        }
    }

    if (out_mat_descriptors.empty())
    {
        std::cerr << "[descriptorComputer][DetectKeyPoints] The Keypoint vector is empty." << std::endl;
        return false;
    }
    
    // Time check end
    end = std::clock();
    double duration = (double)(end-start);
    std::cout << "Description computation duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;

    return true;
}