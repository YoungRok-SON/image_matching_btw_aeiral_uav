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

/* Opencv lib */
#include <opencv4/opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

class ImagePreprocessing
{
public:
    ImagePreprocessing(/* args */);
    ~ImagePreprocessing();

    // UAV Position and Attitude info 
    // For Debugging need to be private
    cv::Point2d m_p2d_uav_lonlat;
    cv::Point2d m_p2d_uav_lonlat_relative;
    cv::Point3d m_p3d_uav_attitude;
    cv::Point2d m_p2d_range_lonlat;

    float m_f_altitude_uav;     // For Debugging need to be private

    cv::Point2i m_p2i_geo_constraints;
private:
    /* Variables */
    bool m_b_initiated = true;

    // UAV - file path and name
    std::string m_str_uav_img_path;
    std::string m_str_uav_img_file_name;
    std::string m_str_uav_img_path_name;
    // Map - file path and name
    std::string m_str_map_path;
    std::string m_str_map_file_name;
    std::string m_str_map_img_path_name;

    // Map Boundary Coordinate - WGS84(Lon(N), Lat(E))
    cv::Point2d m_p2d_top_right_coordinate;
    cv::Point2d m_p2d_top_left_coordinate;
    cv::Point2d m_p2d_bot_right_coordinate;
    cv::Point2d m_p2d_bot_left_coordinate;
    int         m_i_submap_margin; // Margin for rotated uav image.

    // File Metadata
    float m_f_focal_length_uav;
    float m_f_width_image_uav;
    float m_f_height_image_uav;
    float m_f_width_ccd_sensor;
    float m_f_height_ccd_sensor;
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

/* Functios */
private:
    bool init();
    bool ImportImages();
    bool PreprocessImages();
public:
    bool GetImages(cv::Mat &out_uav_img, cv::Mat &out_map_img);
    bool GetCenterOfSubMap(cv::Point2d in_p2d_coordinate_drone, cv::Point2d in_p2d_range_latlon, cv::Point2i in_img_size , cv::Point2i &out_p_submap_center );
    bool GetSubMap(cv::Mat in_img, cv::Point2i in_p2d_submap_center, double in_uav_altitude, cv::Mat &out_submap_img);
    cv::Point2i GetGeographicConstraints(cv::Mat in_submap_img, cv::Mat in_uav_img);
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
    m_str_uav_img_path         = "/home/youngrok/git_ws/image_matching_btw_aeiral_uav/01_uav_images/orthophotos_100m/";
    m_str_uav_img_file_name    = "DJI_0378.JPG";
    m_str_uav_img_path_name    = m_str_uav_img_path + m_str_uav_img_file_name ;
    
    m_str_map_path             = "/home/youngrok/git_ws/image_matching_btw_aeiral_uav/02_map_images/";
    m_str_map_file_name        = "konkuk_latlon_geo_tagged.tif";
    m_str_map_img_path_name    = m_str_map_path + m_str_map_file_name ;

    if(ImportImages() == false)
    {
        std::cerr << "Image import has failed." << std::endl;
        return false;
    }

    m_f_altitude_uav            = 10220.0;          // [m to cm]
    m_f_focal_length_uav        = 0.4 ;             // [mm to cm]
    m_f_width_image_uav         = 4000.0;           // [px]
    m_f_height_image_uav        = 3000.0;           // [px]
    m_f_width_ccd_sensor        = 6.4/10;           // [mm to cm] width  of ccd sensor: check spec of camera.
    m_f_height_ccd_sensor       = 4.8/10;           // [mm to cm] height of ccd sensor: check spec of camera.
    m_f_gsd_uav_img             = m_f_altitude_uav*m_f_width_ccd_sensor/
                                  (m_f_focal_length_uav*m_f_width_image_uav); // [cm]
    m_f_gsd_aerial_map          = 25;                                         // [cm] ground sampling distance: Check the information from the institude of aerial image.
    m_f_resize_factor           = m_f_gsd_uav_img/m_f_gsd_aerial_map;         // resize factor to match gsd of two image
    m_veci_target_size_uav_img  = { int(m_f_width_image_uav * m_f_resize_factor), int(m_f_height_image_uav * m_f_resize_factor)};
    m_f_uav_yaw                 = -130.0;
    m_p2d_uav_lonlat.x = 37.54246977777778; //37.54324908333 
    m_p2d_uav_lonlat.y = 127.07799991666666; //127.0779322777 maybe wrong?

    // x-axis: Lon, y-axis: Lat
    m_p2d_top_right_coordinate.x = 37.5455255124;
    m_p2d_top_right_coordinate.y = 127.0825739840;
    m_p2d_top_left_coordinate.x  = 37.5455290267;
    m_p2d_top_left_coordinate.y  = 127.0744197768;
    m_p2d_bot_right_coordinate.x = 37.5387993632;
    m_p2d_bot_right_coordinate.y = 127.0825667674;
    m_p2d_bot_left_coordinate.x  = 37.5388066215;
    m_p2d_bot_left_coordinate.y  = 127.0744331813;

    // Make this values as positive top - bot, right - left
    m_p2d_range_lonlat.x = m_p2d_top_right_coordinate.x - m_p2d_bot_right_coordinate.x;
    m_p2d_range_lonlat.y = m_p2d_top_right_coordinate.y - m_p2d_top_left_coordinate.y;

    m_p2d_uav_lonlat_relative.x = m_p2d_uav_lonlat.x - m_p2d_bot_left_coordinate.x;
    m_p2d_uav_lonlat_relative.y = m_p2d_uav_lonlat.y - m_p2d_bot_left_coordinate.y;
    m_i_submap_margin = 2;

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
// Output: boolean
bool ImagePreprocessing::ImportImages()
{
    m_mat_uav_img = cv::imread(m_str_uav_img_path_name, cv::IMREAD_COLOR);
    m_mat_map_img = cv::imread(m_str_map_img_path_name, cv::IMREAD_COLOR);

    if(m_mat_uav_img.empty() || m_mat_map_img.empty() )
    {
        std::cerr << "Image import fail." << std::endl;
        return false;
    }
    return true;
}

// Get resize factor using metadata information. Especially GSD value and altitude.
// Input : Nothing
// Output: boolean
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
    
    // Gaussian Blur
	cv::GaussianBlur(m_mat_uav_img, m_mat_uav_img, cv::Size(21,21), 0);

    // Image reszie
    cv::Mat     mat_uav_img_resized;
    cv::Size2d  s2d_target_size(m_veci_target_size_uav_img[0], m_veci_target_size_uav_img[1]);
    cv::resize(m_mat_uav_img,mat_uav_img_resized,s2d_target_size, cv::INTER_LINEAR );

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

// Get center of submap using lat, lon of drone.
// Input : Location of drone(Lon, Lat - WGS84), Image size
// Output: Location of Submap center from UAV Coordinate(WGS84)
bool ImagePreprocessing::GetCenterOfSubMap(cv::Point2d in_p2d_coordinate_drone, cv::Point2d in_p2d_range_latlon, cv::Point2i in_img_size , cv::Point2i &out_p_submap_center )
{
    double d_submap_center_lon = in_img_size.x - (in_p2d_coordinate_drone.x / in_p2d_range_latlon.x * in_img_size.x);  // lon direction is down to up. It should be up to down in image cooridnate.
    double d_submap_center_lat = in_p2d_coordinate_drone.y / in_p2d_range_latlon.y * in_img_size.y;
    // Coordinate conversion: nort(.x) -> image y-axis, east(.y) -> iamge x-axis
    out_p_submap_center.x = (int)d_submap_center_lat; 
    out_p_submap_center.y = (int)d_submap_center_lon;
    
    
    return true;
}


// Get Submap from original img using submap ceneter and size parameter. The submap size is automatically adjusted by UAV altitude.
// Input : Original Image, Submap center(lat,lon-WGS84), uav altitude(cm)
// Output: Submap for image matching
bool ImagePreprocessing::GetSubMap(cv::Mat in_img, cv::Point2i in_p2i_submap_center, double in_uav_altitude, cv::Mat &out_submap_img)
{
    // Define the size of submap around center of submap.
    double d_half_distance_width   = m_f_width_ccd_sensor/2  * in_uav_altitude / m_f_focal_length_uav; // Get distance of Projected area.
    double d_half_distance_height  = m_f_height_ccd_sensor/2 * in_uav_altitude / m_f_focal_length_uav; // the width and height is full width and height.. so make half.
    int    i_half_pixel_num_width  = (int)(d_half_distance_width  / m_f_gsd_aerial_map);              // Get pixel num from distance using gsd of aerial map.
    int    i_half_pixel_num_height = (int)(d_half_distance_height / m_f_gsd_aerial_map);
    // Copy data inside of the boudnary.
    int    i_top_left_x  = in_p2i_submap_center.x - m_i_submap_margin * i_half_pixel_num_width;
    int    i_top_left_y  = in_p2i_submap_center.y - m_i_submap_margin * i_half_pixel_num_height;
    int    i_bot_right_x = in_p2i_submap_center.x + m_i_submap_margin * i_half_pixel_num_width;
    int    i_bot_right_y = in_p2i_submap_center.y + m_i_submap_margin * i_half_pixel_num_height;

    out_submap_img = in_img(cv::Rect( cv::Point(i_top_left_x, i_top_left_y) ,  cv::Point(i_bot_right_x, i_bot_right_y)));
    return true;
}

cv::Point2i ImagePreprocessing::GetGeographicConstraints(cv::Mat in_submap_img, cv::Mat in_uav_img) // 이걸 중심점을 어떻게 구해놓을지 ... 다른 함수에서 구해서 멤버로 넣어 놓을지, 아니면 여기서 다시 연산을 할지?
{
    m_p2i_geo_constraints.x = in_submap_img.cols;
    m_p2i_geo_constraints.y = in_submap_img.rows;

    return m_p2i_geo_constraints;
}

#endif //__IMAGE_PREPROCESSING__