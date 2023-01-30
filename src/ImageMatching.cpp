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
    return true;
}

// Extract Key point and descriptor from image with Variable using SLIC boudnary and BRISK algorithm.
// Input : Image to extrac key point, SLIC parameters.
// Output: Keypoint Set vector.
bool ImageMatching::ComputeKeyDescriptorSlic(cv::Mat in_img, ImgType in_img_type, bool do_downsample, DescriptorType in_descriptor_type, std::vector<cv::KeyPoint> &out_vec_key_points, cv::Mat &out_mat_descriptors )
{
    // 파라미터를 클래스 안쪽에 변수로 두는게 나을까 아니면 메인함수에서 가지고 있는게 나을까
    // 이 변수는 클래스 안에 선언하고 이미지가 어떤것 인지만 알면되니까 이넘으로 넘겨주는게 나을듯?
    cv::Mat mat_slic_result, mat_slic_mask;
    int i_width, i_height, i_channels = in_img.channels();
    int i_display_mode = 0;
    cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
    cv::Mat mat_gray_img;
    i_width  = in_img.size().width;
    i_height = in_img.size().height;
    std::clock_t start, end;
    double duration;

    if( in_img_type == UAV)
    {
        start = std::clock();

        cv::cvtColor(in_img, mat_gray_img, cv::COLOR_BGR2GRAY);

        // Generate Initial SEEDs for SLIC.
        seeds = cv::ximgproc::createSuperpixelSEEDS(i_width           , i_height     , i_channels         , m_i_uav_num_superpixels,
                                                    m_i_uav_num_levels, m_i_uav_prior, m_i_uav_num_histogram_bins, m_b_uav_double_step    );
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

    }else if (in_img_type == MAP)
    {
        start = std::clock();

        cv::cvtColor(in_img, mat_gray_img, cv::COLOR_BGR2GRAY);

        // Generate Initial SEEDs for SLIC.
        seeds = cv::ximgproc::createSuperpixelSEEDS(i_width           , i_height     , i_channels         , m_i_map_num_superpixels,
                                                    m_i_map_num_levels, m_i_map_prior, m_i_map_num_histogram_bins, m_b_map_double_step    );
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
    }
        // Extacto Key points using SLIC Boundary mask and image
        std::vector<cv::KeyPoint> vec_key_points;
        ExtractKeypoints(mat_gray_img, mat_slic_mask, vec_key_points);
        out_vec_key_points = vec_key_points;
        std::cout << "Number of Key points from SLIC: " << vec_key_points.size() << std::endl;

        // Compute descriptor from vector of keypoints.
        if( in_descriptor_type == BRISK)
        {
            cv::Ptr<cv::BRISK> brisk_feature = cv::BRISK::create(m_i_threshold, m_i_octaves, m_f_pattern_scale);
            brisk_feature->compute(in_img, vec_key_points, out_mat_descriptors);
        }else if( in_descriptor_type == SIFT )
        {
            cv::Ptr<cv::SIFT> ptr_sift_feature = cv::SIFT::create(m_i_n_feature, m_i_n_octav_layers, m_d_contrast_th, m_d_edge_th, m_d_sigma);
            ptr_sift_feature->compute(in_img, vec_key_points,out_mat_descriptors);
        }
        std::cout << "matrix type: " << out_mat_descriptors.type() << std::endl;
        
        // Time check for SLIC algorithm.
        end = std::clock();
        duration = (double)(end-start);
        std::cout << "SLICK based keypoint and descriptor extraction duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;

        return true;
    return false;
}

bool ImageMatching::MatchImages(cv::Mat in_uav_img, cv::Mat in_map_img, 
                                std::vector<cv::KeyPoint> in_vec_uav_keypoint, std::vector<cv::KeyPoint> in_vec_map_keypoint,
                                cv::Mat in_mat_uav_descriptors, cv::Mat in_mat_map_descriptors, DescriptorType in_descriptor_type)
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
    std::vector<std::vector<cv::DMatch>> vvec_knn_matches;
    if (in_descriptor_type == BRISK)
    {
    cv::FlannBasedMatcher obj_matcher = cv::FlannBasedMatcher(  cv::makePtr<cv::flann::LshIndexParams>(6,12,1),
                                                                cv::makePtr<cv::flann::SearchParams>(50)      );
    obj_matcher.knnMatch(in_mat_uav_descriptors, in_mat_map_descriptors, vvec_knn_matches, m_i_num_one_to_many);
    }
    else if (in_descriptor_type == SIFT)
    {
        cv::Ptr<cv::DescriptorMatcher> ptr_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        ptr_matcher->knnMatch(in_mat_uav_descriptors, in_mat_map_descriptors,vvec_knn_matches, m_i_num_one_to_many);
    }
    end = std::clock();
    duration = (double)(end-start);
    std::cout << "FLANN-based featue matching duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;
    std::cout << "UAV keypoint from mache result: " << vvec_knn_matches.size() << std::endl;
    std::cout << "Map keypoint from mache result: " << vvec_knn_matches[0].size() << std::endl;
    // Histogram voting using geographical(지리적) infomation.
    cv::Point p_delta_translation;
    GetCenterOfGeographyConstraint(vvec_knn_matches, in_vec_uav_keypoint, in_vec_map_keypoint, p_delta_translation);
    // Matching refinement using Template matching with Matching cadidates.
    std::vector<cv::DMatch> vec_refined_mathing_result;
    RefineMatchedResult(in_uav_img, in_map_img, in_vec_uav_keypoint, in_vec_map_keypoint, p_delta_translation, vec_refined_mathing_result);
    return true;
}

bool ImageMatching::RefineMatchedResult( cv::Mat in_uav_img, cv::Mat in_map_img, 
                                         std::vector<cv::KeyPoint> in_vec_uav_keypoints, std::vector<cv::KeyPoint> in_vec_map_keypoints,
                                         cv::Point in_translation,
                                         std::vector<cv::DMatch> out_refined_matching_result)
{
    std::clock_t start, end;
    double duration;
    start = std::clock();

    /* Generate interest keypoint set */
    // Image boundary check using geographical constraint.
    int i_left_boundary, i_right_boundary, i_top_boundary, i_bot_boundary;
    i_left_boundary  = - in_translation.x;
    i_right_boundary =   in_uav_img.cols + i_left_boundary ;
    i_top_boundary   = - in_translation.y;
    i_bot_boundary   =   in_uav_img.rows + i_top_boundary;
    if( i_left_boundary < 0 || i_right_boundary > in_map_img.cols-1 || i_top_boundary < 0 || i_bot_boundary > in_map_img.rows-1)
    {
        std::cerr << "UAV Image mathing result exceeds the map image boundary." << std::endl;
        return false;
        std::cout << i_left_boundary <<"  "<< i_right_boundary <<"  "<< i_top_boundary <<"  "<< i_bot_boundary << std::endl;
    }

    // Get interest keypoints from whole key point vector.
    std::vector<cv::Point2f> vecp_interest_keypoints_loc;
    for ( int idx_keypoint_map = 0; idx_keypoint_map < in_vec_map_keypoints.size();  idx_keypoint_map++)
    {
        int i_keypoint_x = in_vec_map_keypoints[idx_keypoint_map].pt.x;
        int i_keypoint_y = in_vec_map_keypoints[idx_keypoint_map].pt.y;
        
        if ( i_left_boundary <= i_keypoint_x && i_keypoint_x <= i_right_boundary && i_top_boundary  <= i_keypoint_y && i_keypoint_y <= i_bot_boundary )
        {
            vecp_interest_keypoints_loc.push_back(cv::Point2f(i_keypoint_x, i_keypoint_y));
        }
    }
    
    if ( vecp_interest_keypoints_loc.empty() == true )
    {
        std::cerr << "interest boundary is empty." << std::endl;
        return false;
    }

    // Generate KD-tree

    cv::flann::KDTreeIndexParams kdtree_index_params;
    cv::flann::Index kdtree (cv::Mat(vecp_interest_keypoints_loc).reshape(1), kdtree_index_params);
    
    // Search and push back to vector
    for (size_t idx_query = 0; idx_query < in_vec_uav_keypoints.size(); idx_query++)
    {
        std::vector<float> query{in_vec_uav_keypoints[idx_query].pt.x, in_vec_uav_keypoints[idx_query].pt.y};
        std::vector<int> veci_close_indices;
        std::vector<float> vecf_distance;
        kdtree.radiusSearch(query, veci_close_indices, vecf_distance, 20.0, 50, cv::flann::SearchParams(32));
        /// Check if there is noting close near query.
        if(veci_close_indices.empty() == true)
        {
            std::cerr << "[RefineMatchedResult] There is no keypoint near query point." << std::endl;
            continue;
        }
        if ( b_dbg_vis_close_key_point_extract == true )
        {
            
        }
    }

    /* Refine matching result using Template matching */
    // Get Base image (include all candidate keypoints)

    // Get score and save into container

    // Get Best matching and save into container
    
    end = std::clock();
    duration = (double)(end-start);
    std::cout << "Template matching based refinement duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;
    return true;
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
                cv::KeyPoint slic_key_point(slick_point, m_i_keypoint_size, m_f_keypoint_angle);
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

bool ImageMatching::GetCenterOfGeographyConstraint( std::vector<std::vector<cv::DMatch>> in_vvec_dmatch_reuslt,
                                                    std::vector<cv::KeyPoint> in_vec_uav_keypoints,
                                                    std::vector<cv::KeyPoint> in_vec_map_keypoints,
                                                    cv::Point &out_center_location)
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
    std::vector<std::pair<int,int>> vpii_mode_x; // <bin's center, number of element> vpii: vector pair int int, mode: most frequent value
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
            // std::cout << "nuber of bin: "       <<  i_count_boundary 
            //           << " number of element: " <<  i_num_element_in_bin
            //           << " Center value x: "    <<  i_center_value_x << std::endl;
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

    // For debug
    m_p2i_trans.x = i_mode_x;
    m_p2i_trans.y = i_mode_y;
    
    std::cout << "Mode value of X pixel difference: " << i_mode_x << std::endl;
    std::cout << "How many times?: " << vpii_mode_x[0].second << std::endl;
    std::cout << "Mode value of Y pixel difference: " << i_mode_y << std::endl;
    std::cout << "How many times?: " << vpii_mode_y[0].second << std::endl;
    std::cout << "Histogram filtering duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;
    out_center_location.x = i_mode_x;
    out_center_location.y = i_mode_y;

    end = std::clock();
    duration = (double)(end-start);
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
