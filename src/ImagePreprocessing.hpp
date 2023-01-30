/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file ImagePreprocessing.hpp
 * @brief Image Import and get meta data from image
 * @version 1.0
 * @date 1-11-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __IMAGE_PREPROCESSING__
#define __IMAGE_PREPROCESSING__

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class ImagePreprocessing
{
public:
    ImagePreprocessing(/* args */);
    ~ImagePreprocessing();
private:
    /* Variables */
    bool m_b_initiated = true;

    // UAV - file path and name
    std::string m_str_uav_img_path         = "../01_uav_images/orthophotos_100m/";
    std::string m_str_uav_img_file_name    = "DJI_0378.JPG";
    std::string m_str_uav_img_path_name    = m_str_uav_img_path + m_str_uav_img_file_name ;
    // Map - file path and name
    std::string m_str_map_path         = "../02_map_images/";
    std::string m_str_map_file_name    = "aerial_map_konkuk_25cm.png";
    std::string m_str_map_img_path_name    = m_str_map_path + m_str_map_file_name ;

    // File Metadata
    float m_f_altitude_uav;     
    float m_f_focal_length_uav;
    float m_f_width_image_uav;
    float m_f_height_image_uav;
    float m_f_width_ccd_sensor;
    float m_f_gsd_uav_img;   
    float m_f_gsd_aerial_map;
    float m_f_resize_factor;
    std::vector<int> m_veci_target_size_uav_img;
    float m_f_uav_yaw;
    // Import Images
    cv::Mat m_mat_uav_img;
    cv::Mat m_mat_map_img;
    cv::Mat m_mat_uav_img_preprocessed;
    cv::Mat m_mat_map_img_preprocessed;

private:
    /* Functios */
    bool init();
    bool ImportImages();
    bool PreprocessImages();
public:
    bool GetImages(cv::Mat &out_uav_img, cv::Mat &out_map_img);

};

ImagePreprocessing::ImagePreprocessing(/* args */)
{
    if( init() == false )
    {
        m_b_initiated = false;
        std::cerr << "ImagePreprocessing Class has failed to initiate." << std::endl;
    }
}

ImagePreprocessing::~ImagePreprocessing()
{
}

bool ImagePreprocessing::init()
{
    if(ImportImages() == false)
    {
        std::cerr << "Image import has failed." << std::endl;
        return false;
    }
    m_f_altitude_uav            = 10220.0;          // [m to cm]
    m_f_focal_length_uav        = 0.45 ;            // [mm to cm]
    m_f_width_image_uav         = 4000.0;           // [px]
    m_f_height_image_uav        = 3000.0;           // [px]
    m_f_width_ccd_sensor        = 6.4/10;           // [mm to cm] width of ccd sensor: check spec of camera.
    m_f_gsd_uav_img             = m_f_altitude_uav*m_f_width_ccd_sensor/
                                  (m_f_focal_length_uav*m_f_width_image_uav); // [cm]
    m_f_gsd_aerial_map          = 25;                                         // [cm] ground sampling distance: Check the information from the institude of aerial image.
    m_f_resize_factor           = m_f_gsd_uav_img/m_f_gsd_aerial_map;         // resize factor to match gsd of two image
    m_veci_target_size_uav_img  = { int(m_f_width_image_uav * m_f_resize_factor), int(m_f_height_image_uav * m_f_resize_factor)};
    m_f_uav_yaw                 = -130.0;
    std::cout << "UAV Altitude         : " << m_f_altitude_uav << std::endl;
    std::cout << "UAV focal length     : " << m_f_focal_length_uav << std::endl;
    std::cout << "UAV image width      : " << m_f_width_image_uav << std::endl;
    std::cout << "UAV ccd width        : " << m_f_width_ccd_sensor << std::endl;
    std::cout << "UAV image GSD        : " << m_f_gsd_uav_img << std::endl;
    std::cout << "Map image GSD        : " << m_f_gsd_aerial_map << std::endl;
    std::cout << "UAV resize ratio     : " << m_f_resize_factor << std::endl;
    std::cout << "UAV target image size: w-" << m_veci_target_size_uav_img[0] << ", H- "<< m_veci_target_size_uav_img[1] << std::endl;

    return true;
}

// Import Image from given path of ImagePreprocessing.hpp
// Input: noting
// output: boolean
bool ImagePreprocessing::ImportImages()
{
    m_mat_uav_img = cv::imread(m_str_uav_img_path_name, cv::IMREAD_COLOR);
    m_mat_map_img = cv::imread(m_str_map_img_path_name, cv::IMREAD_COLOR);

    if(m_mat_uav_img.empty() || m_mat_map_img.empty())
    {
        std::cerr << "Image import fail." << std::endl;
        return false;
    }
    return true;
}

// Get resize factor using metadata information. Especially GSD value and altitude.
// Input: noting
// output: boolean
bool ImagePreprocessing::PreprocessImages()
{
    if( m_b_initiated == false)
    {
        std::cerr << "ImagePreprocessing Class has failed to initiate." << std::endl;
        return false;
    }
    if(m_veci_target_size_uav_img.empty() == true)
    {
        std::cout << "Image target size is not determined." << std::endl;
        return false;
    }

    // Image reszie
    cv::Mat     mat_uav_img_resized;
    cv::Size2d  s2d_target_size(m_veci_target_size_uav_img[0], m_veci_target_size_uav_img[1]);
    cv::resize(m_mat_uav_img,mat_uav_img_resized,s2d_target_size,cv::INTER_LINEAR);

    // Image Rotation Alignment
    cv::Mat mat_uav_image_rotated;
    float   f_target_rotate_angle = m_f_uav_yaw;
    cv::Point2f p2f_center_of_img( (mat_uav_img_resized.cols-1)/2.0, (mat_uav_img_resized.rows-1)/2.0 );
    /// Get Center of image
    cv::Mat mat_rotation_matrix = cv::getRotationMatrix2D(p2f_center_of_img, f_target_rotate_angle, 1.0);
    /// Determine the bounding box size
    cv::Rect2f r2f_bounding_box = cv::RotatedRect(cv::Point2f(), mat_uav_img_resized.size(), f_target_rotate_angle).boundingRect2f();
    /// Adjust rotation matrix
    mat_rotation_matrix.at<double>(0,2) += r2f_bounding_box.width/2.0 - mat_uav_img_resized.cols/2.0;
    mat_rotation_matrix.at<double>(1,2) += r2f_bounding_box.height/2.0 - mat_uav_img_resized.rows/2.0;

    cv::warpAffine(mat_uav_img_resized, mat_uav_image_rotated,mat_rotation_matrix,r2f_bounding_box.size());
    mat_uav_image_rotated.copyTo(m_mat_uav_img_preprocessed);   
    return true;
}

// Pass the image set
// Input: noting
// output: boolean
bool ImagePreprocessing::GetImages(cv::Mat &out_uav_img, cv::Mat &out_map_img)
{
    if( m_b_initiated == false)
    {
        std::cerr << "ImagePreprocessing Class has failed to initiate." << std::endl;
        return false;
    }
    if( PreprocessImages() == false )
    {
        std::cerr << "ImagePreprocessing has failed." << std::endl;
        return false;
    }   
    
    out_uav_img = m_mat_uav_img_preprocessed;
    out_map_img = m_mat_map_img;
    return true;
}

#endif //__IMAGE_PREPROCESSING__