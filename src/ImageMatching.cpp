/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file ImageMatching.cpp
 * @brief Match images btween UAV(low altitude) with Aerial Image Map(Hight Altitude)
 * @version 1.0
 * @date 1-11-2023
 * @bug No known bugs
 * @warning No warnings
 */

#include "ImageMatching.hpp"

ImageMatching::ImageMatching()
{
    if( Init() == false )
    {
        std::cout  << "Image Matching Class Init has failed." << std::endl;
    }

    m_b_initiated = true;
}

ImageMatching::~ImageMatching() {}

bool ImageMatching::Init()
{
    return true;
}

// Run function of Image Matching Class.
// Input : UAV and Map image to match.
// Output: Homography matrix for matching Image.
bool ImageMatching::RunImageMatching()
{
    if( m_b_initiated == false)
    {
        std::cerr << "Image Matching class initiation is failed." << std::endl;
        return false;
    }
    // TBD after all functions are defiend.
    // if (in_img.empty() == true )
    // {
    // std::cerr << "Input image is empty." << std::endl;
    // return false;
    // }
}

// Extract Key point and descriptor from image with Variable using SLIC boudnary and BRISK algorithm.
// Input : Image to extrac key point, SLIC parameters.
// Output: Keypoint Set vector.
bool ImageMatching::ComputeKeyDescriptorSlicBrisk(cv::Mat in_img, ImgType in_img_type, bool do_downsample, std::vector<cv::KeyPoint> &out_vec_key_points, cv::Mat &out_mat_descriptors )
{
    // 파라미터를 클래스 안쪽에 변수로 두는게 나을까 아니면 메인함수에서 가지고 있는게 나을까
    // 이 변수는 클래스 안에 선언하고 이미지가 어떤것 인지만 알면되니까 이넘으로 넘겨주는게 나을듯?
    cv::Mat mat_slic_result, mat_slic_mask;
    if( in_img_type == UAV)
    {
        cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
        int i_width, i_height;
        int i_display_mode = 0;
        
        i_width = in_img.size().width;
        i_height = in_img.size().height;
        // Generate Initial SEEDs for SLIC.
        seeds = cv::ximgproc::createSuperpixelSEEDS(i_width,i_height, in_img.channels(), m_i_uav_num_superpixels,
                m_i_uav_num_levels, m_i_uav_prior, m_i_uav_num_histogram_bins, m_b_uav_double_step);
        // Convert img type
        cv::Mat mat_converted_img;
        cv::cvtColor(in_img, mat_converted_img, cv::COLOR_BGR2HSV);

        double time = (double) cv::getTickCount();

        // Calculates the superpixel sgmentation on a given image with parameters.
        seeds->iterate(mat_converted_img,m_i_uav_num_iterations);
        mat_slic_result = in_img;

        // Time check for SLIC algorithm.
        time =  ( (double) cv::getTickCount() - time ) / cv::getTickFrequency();
        std::printf("SEEDs segmentation took %i ms with %3i superpixels. \n",
                    (int) (time*1000), seeds->getNumberOfSuperpixels());

        // Get the segmented result.
        cv::Mat mat_labels;
        seeds->getLabels(mat_labels);

        // Get contour to use as key point.
        seeds->getLabelContourMask(mat_slic_mask,false);

        // Visualize SLIC result boundary.
        mat_slic_result.setTo(cv::Scalar(0,0,255), mat_slic_mask);
        cv::imshow("SLIC result of UAV Image", mat_slic_result);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // Get Key points from SLIC boudnary.
        std::vector<cv::KeyPoint> vec_uav_key_points;
        ExtractKeypoints(mat_slic_mask, vec_uav_key_points);
        std::cout << "Number of Key points from SLIC: " << vec_uav_key_points.size() << std::endl;

        // Compute descriptor from vector of keypoints.
        cv::Mat mat_descriptor;
        cv::Ptr<cv::BRISK> brisk_feature = cv::BRISK::create(m_i_threshold, m_i_octaves, m_f_pattern_scale);
        brisk_feature->compute(in_img, vec_uav_key_points, mat_descriptor);

        return true;
    }

    return false;
}

bool ImageMatching::ExtractKeypoints(cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint>  &out_vector_keypoints)
{
    for(int idx_y = 0; idx_y < in_mat_slic_mask.rows; idx_y++)
    {
        const unsigned char* Mi = in_mat_slic_mask.ptr<unsigned char>(idx_y);
        for(int idx_x = 0; idx_x < in_mat_slic_mask.cols; idx_x++)
        {
            int element = Mi[idx_x];
            if ( element == 255 ) // SLIC Boundary output is 255 which is max of uchar. but why?
            {
                cv::KeyPoint slic_key_point(idx_x,idx_y,5);  // size=5 is magic number.
                out_vector_keypoints.push_back(slic_key_point);
            }
        }
    }

}



bool ImageMatching::MatchImages()
{

}