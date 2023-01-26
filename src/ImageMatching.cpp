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
**/

#include "ImageMatching.hpp"

bool comp(const std::pair<int, int> &p1,const std::pair<int, int> &p2){
    if(p1.second == p2.second){     //빈도수가 같으면 
        return p1.first < p2.first; //숫자(key)작은게 앞으로 
    }
    return p1.second > p2.second;    //다르면 빈도수가 큰게 앞으로 
}

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
        std::clock_t start, end;
        double duration;
        start = std::clock();

        cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
        int i_width, i_height;
        int i_display_mode = 0;
        cv::Mat mat_gray_img;
        cv::cvtColor(in_img, mat_gray_img, cv::COLOR_BGR2GRAY);

        i_width = in_img.size().width;
        i_height = in_img.size().height;
        // Generate Initial SEEDs for SLIC.
        seeds = cv::ximgproc::createSuperpixelSEEDS(i_width,i_height, in_img.channels(), m_i_uav_num_superpixels,
                m_i_uav_num_levels, m_i_uav_prior, m_i_uav_num_histogram_bins, m_b_uav_double_step);
        // Convert img type
        cv::Mat mat_converted_img;
        cv::cvtColor(in_img, mat_converted_img, cv::COLOR_BGR2HSV);

        // Calculates the superpixel sgmentation on a given image with parameters.
        seeds->iterate(mat_converted_img,m_i_uav_num_iterations);
        mat_slic_result = in_img;

        // Get the segmented result.
        cv::Mat mat_labels;
        seeds->getLabels(mat_labels);

        // Get contour to use as key point.
        seeds->getLabelContourMask(mat_slic_mask,false);
        // Extacto Key points using SLIC Boundary mask and image
        std::vector<cv::KeyPoint> vec_uav_key_points;
        ExtractKeypoints(mat_gray_img, mat_slic_mask, vec_uav_key_points);
        out_vec_key_points = vec_uav_key_points;
        std::cout << "Number of Key points from SLIC: " << vec_uav_key_points.size() << std::endl;

        // Compute descriptor from vector of keypoints.
        cv::Mat mat_descriptor;
        cv::Ptr<cv::BRISK> brisk_feature = cv::BRISK::create(m_i_threshold, m_i_octaves, m_f_pattern_scale);
        brisk_feature->compute(in_img, vec_uav_key_points, mat_descriptor);
        mat_descriptor.copyTo(out_mat_descriptors);

        std::cout << "matrix type: " << mat_descriptor.type() << std::endl;
        
        // Time check for SLIC algorithm.
        end = std::clock();
        duration = (double)(end-start);
        std::cout << "SLICK based keypoint and descriptor extraction duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;

        return true;

    }else if (in_img_type == MAP)
    {
        // TODO
        // 
    }

    return false;
}

bool ImageMatching::ExtractKeypoints(cv::Mat in_mat_gray_img, cv::Mat in_mat_slic_mask, std::vector<cv::KeyPoint>  &out_vector_keypoints)
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
                 i_bottom_pixel != 0      ) // SLIC Boundary output is 255 which is max of uchar. but why?
            {
                // Generate Key point object and add to vector.
                cv::Point2f slick_point(idx_x, idx_y);
                cv::KeyPoint slic_key_point(slick_point,m_i_keypoint_size,0.0);
                out_vector_keypoints.push_back(slic_key_point);
            }
        }
    }
}



bool ImageMatching::MatchImages(std::vector<cv::KeyPoint> in_vec_uav_keypoint, std::vector<cv::KeyPoint> in_vec_map_keypoint,
                                cv::Mat in_mat_uav_descriptors, cv::Mat in_mat_map_descriptors)
{
    if( in_vec_uav_keypoint.size() < m_i_num_keypoint_threshold)
    {
        std::cerr << "UAV Key points are too few." << std::endl;
        return false;

    }if( in_vec_map_keypoint.size() < m_i_num_keypoint_threshold)
    {
        std::cerr << "Map Key points are too few." << std::endl;
        return false;
    } if( in_mat_uav_descriptors.size().height < m_i_num_keypoint_threshold)
    {
        std::cerr << "UAV Descriptor  are too few." << std::endl;
        return false;
    } if( in_mat_map_descriptors.size().height < m_i_num_keypoint_threshold)
    {
        std::cerr << "Map Descriptor  are too few." << std::endl;
        return false;
    }
    std::clock_t start, end;
    double duration;

    start = std::clock();
    // Descriptor matching
    cv::FlannBasedMatcher obj_matcher = cv::FlannBasedMatcher(  cv::makePtr<cv::flann::LshIndexParams>(6,12,1),
                                                                cv::makePtr<cv::flann::SearchParams>(50)      );

    std::vector<std::vector<cv::DMatch>> vvec_knn_matches;
    obj_matcher.knnMatch(in_mat_uav_descriptors, in_mat_map_descriptors, vvec_knn_matches, m_i_num_one_to_many);
    
    end = std::clock();
    duration = (double)(end-start);
    std::cout << "FLANN-based featue matching duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;
    std::cout << "UAV keypoint from mache result: " << vvec_knn_matches.size() << std::endl;
    std::cout << "Map keypoint from mache result: " << vvec_knn_matches[0].size() << std::endl;
    // Histogram voting using geographical(지리적) infomation.
    cv::Point2f p2f_center_point;
    GetCenterOfGeographyConstraint(vvec_knn_matches,in_vec_uav_keypoint, in_vec_map_keypoint, p2f_center_point);
    // Matching refinement using Template matching with Matching cadidates.

    return true;
}

bool ImageMatching::GetCenterOfGeographyConstraint( std::vector<std::vector<cv::DMatch>> in_vvec_dmatch_reuslt,
                                                    std::vector<cv::KeyPoint> in_vec_uav_keypoints,
                                                    std::vector<cv::KeyPoint> in_vec_map_keypoints,
                                                    cv::Point2f &out_center_location)
{
    if( in_vvec_dmatch_reuslt.size() < m_i_keypoint_size)
    {
        std::cerr << "Datching result is too few." << std::endl;
        return false;
    }
    std::clock_t start, end;
    double duration;
    start = std::clock();
    std::cout << "matches size Idx: " << in_vvec_dmatch_reuslt.size() << std::endl;

    // Set matching reuslt as mat type.
    int i_total_size = in_vvec_dmatch_reuslt.size() * in_vvec_dmatch_reuslt[0].size();
    std::vector<int> vec_pixel_diff_x(i_total_size,0), vec_pixel_diff_y(i_total_size, 0);
    // Get difference between uav image pixel and map image pixel at match result.
    for (int idx_uav = 0; idx_uav < in_vvec_dmatch_reuslt.size(); idx_uav++)
    {
        int i_query_x    = in_vec_uav_keypoints [idx_uav].pt.x;
        int i_query_y    = in_vec_uav_keypoints [idx_uav].pt.y;
        for (int idx_map = 0; idx_map < m_i_num_one_to_many; idx_map++)
        {
            int i_idx_train  = in_vvec_dmatch_reuslt[idx_uav][idx_map].trainIdx;
            int i_train_x    = in_vec_map_keypoints [i_idx_train].pt.x;
            int i_train_y    = in_vec_map_keypoints [i_idx_train].pt.y;
            int i_diff_x     = i_query_x - i_train_x;
            int i_diff_y     = i_query_y - i_train_y;
            vec_pixel_diff_x[idx_uav * m_i_num_one_to_many + idx_map] = i_diff_x;
            vec_pixel_diff_y[idx_uav * m_i_num_one_to_many + idx_map] = i_diff_y;
        }
    }
    
    // Sort vector as accending.
    std::sort(vec_pixel_diff_x.begin(), vec_pixel_diff_x.end());
    std::sort(vec_pixel_diff_y.begin(), vec_pixel_diff_y.end());
    int i_diff_x_min = vec_pixel_diff_x.front(), i_diff_x_max = vec_pixel_diff_x.back(), 
        i_diff_y_min = vec_pixel_diff_y.front(), i_diff_y_max = vec_pixel_diff_y.back();

    // Get full range of histogram
    int i_hist_range_x  = i_diff_x_max - i_diff_x_min;
    int i_hist_range_y  = i_diff_y_max - i_diff_y_min;

    // Get Section boudnary for histogram
    int i_num_section_x = std::ceil(i_hist_range_x / m_i_num_pixels_in_bin);
    int i_num_section_y = std::ceil(i_hist_range_y / m_i_num_pixels_in_bin);
    std::vector<int> veci_boundary_x;
    std::vector<int> veci_boundary_y;

    // Get section boundary vector
    for (int idx_section = 0; idx_section <= i_num_section_x; idx_section++){veci_boundary_x.push_back(i_diff_x_min + m_i_num_pixels_in_bin * idx_section); }
    for (int idx_section = 0; idx_section <= i_num_section_y; idx_section++){veci_boundary_y.push_back(i_diff_y_min + m_i_num_pixels_in_bin * idx_section); }

    // Generate vector of pair that has mode infomation.
    std::vector<std::pair<int,int>> vpii_mode_x; // <bin's center, number of element>
    std::vector<std::pair<int,int>> vpii_mode_y; 
    
    // 하나씩 돌아가면서 범위 안에 있다면 수를 증가 시키는 방식으로 변환
    vpii_mode_x.push_back( std::pair<int,int>( (veci_boundary_x[0]+veci_boundary_x[1])/2 , 1) );
    vpii_mode_y.push_back( std::pair<int,int>( (veci_boundary_y[0]+veci_boundary_y[1])/2 , 1) );

    // Get X-axis Histogram
    int i_count_boundary = 0;
    for( int idx = 0; idx < vec_pixel_diff_x.size(); idx++)
    {
        if( vec_pixel_diff_x[idx] >= veci_boundary_x[i_count_boundary] &&
            vec_pixel_diff_x[idx] < veci_boundary_x[i_count_boundary + 1] )
        {
            std::pair<int,int> tmp = vpii_mode_x.back();
            tmp.second++;
            vpii_mode_x.pop_back();
            vpii_mode_x.push_back(tmp);
        }else
        {   
            i_count_boundary += 1;
            int i_num_element_in_bin = vpii_mode_x.back().second;
            int i_center_value_x     = vpii_mode_x.back().first;
            std::cout << "nuber of bin: "       <<  i_count_boundary 
                      << " number of element: " <<  i_num_element_in_bin
                      << " Center value x: "    <<  i_center_value_x << std::endl;
            vpii_mode_x.push_back(std::pair<int,int>( std::ceil(veci_boundary_x[i_count_boundary]+veci_boundary_x[i_count_boundary+1])/2 , 1) );
        }
    }

    // Get Y-axis Histogram
    i_count_boundary = 0;
    for( int idx = 0; idx < vec_pixel_diff_y.size(); idx++)
    {
        if( vec_pixel_diff_y[idx] >= veci_boundary_y[i_count_boundary] &&
            vec_pixel_diff_y[idx] < veci_boundary_y[i_count_boundary + 1] )
        {
            std::pair<int,int> tmp = vpii_mode_y.back();
            tmp.second++;
            vpii_mode_y.pop_back();
            vpii_mode_y.push_back(tmp);
        }else
        {   
            i_count_boundary += 1;
            int i_num_element_in_bin = vpii_mode_y.back().second;
            int i_center_value_y     = vpii_mode_y.back().first;
            // std::cout << "nuber of bin: "       <<  i_count_boundary 
            //           << " number of element: " <<  i_num_element_in_bin
            //           << " Center value y: "    <<  i_center_value_y << std::endl;
            vpii_mode_y.push_back(std::pair<int,int>( std::ceil(veci_boundary_y[i_count_boundary]+veci_boundary_y[i_count_boundary+1])/2 , 1) );
        }
    }

    // Find mode of difference of pixel.
    int i_mode_x = 0, i_mode_y = 0;
    sort(vpii_mode_x.begin(), vpii_mode_x.end(), comp);
    sort(vpii_mode_y.begin(), vpii_mode_y.end(), comp);
    
    i_mode_x = vpii_mode_x[0].first;
    i_mode_y = vpii_mode_y[0].first;

    std::cout << "Mode value of X pixel difference: " << i_mode_x << std::endl;
    std::cout << "How many times?: " << vpii_mode_x[0].second << std::endl;
    std::cout << "Mode value of Y pixel difference: " << i_mode_y << std::endl;
    std::cout << "How many times?: " << vpii_mode_y[0].second << std::endl;

    end = std::clock();
    duration = (double)(end-start);
    std::cout << "Histogram filtering duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;
    out_center_location.x = i_mode_x;
    out_center_location.y = i_mode_y;
    return true;
}


void ImageMatching::ShowKeypoints(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_key_points, cv::Mat &out_mat_keypoint_img)
{
    in_img.copyTo(out_mat_keypoint_img);
	std::cout << "UAV - Number of Key points: " << in_vec_key_points.size()  << std::endl;
	for (int idx = 0; idx < in_vec_key_points.size(); idx ++)
	{
		int point_x = in_vec_key_points[idx].pt.x;
		int point_y = in_vec_key_points[idx].pt.y;
		cv::circle(out_mat_keypoint_img, cv::Point(point_x, point_y), 1, cv::Scalar(100,255,100), -1);
	}
}
