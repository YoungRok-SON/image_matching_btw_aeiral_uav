/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file descriptorComputer.hpp
 * @brief Compute descriptor using various method
 * @version 1.0
 * @date 03-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __DESCRIPTOR_COMPUTER__
#define __DESCRIPTOR_COMPUTER__

#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/xfeatures2d.hpp"

#include "utility.hpp"

class descriptorComputer
{
private:
/* Member Variables */
    // Key point extraction parameters
    int    m_i_keypoint_size;
    int    m_f_keypoint_angle;
    // for slic-based SIFT Keypoints 
    int    m_i_n_feature;
    int    m_i_n_octav_layers;
    double m_d_contrast_th;
    double m_d_edge_th;
    double m_d_sigma;
    
    bool init();
public:
/* Init Functions */
    descriptorComputer(/* args */);
    ~descriptorComputer();
public:
/* Member Functions */
    bool ComputeDescriptors(cv::Mat                   in_img, 
                            std::vector<cv::KeyPoint> in_veckey_keypoitns, 
                            KeyDescType               in_desc_type, 
                            cv::Mat&                  out_mat_descriptors);
};


#endif //__DESCRIPTOR_COMPUTER__