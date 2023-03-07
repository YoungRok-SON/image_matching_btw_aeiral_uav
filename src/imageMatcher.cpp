
/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file imageMatcher.cpp
 * @brief Match two image using various method.
 * @version 1.0
 * @date 07-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#include "../include/imageMatcher.hpp"


imageMatcher::imageMatcher()
{
    if( init() == false )
    {
        std::cout  << "[keyPointDetector] Class init has failed." << std::endl;
    }
    m_b_initiated = true;
}

imageMatcher::~imageMatcher(){}

bool imageMatcher::init()
{
    m_b_show_match_result = true;
    m_b_do_ratio_test = true; 
    m_b_show_keypoint_result = false;
    
    // For Ratio test
    m_f_ratio_th  = 0.9;
    // For SLIC-based Matching
    m_i_num_one_to_many        = 500; 
    m_i_num_keypoint_threshold = 1000;
    // Histogram voting for geopraphical constriant
    m_i_num_pixels_in_bin      = 1;
    cv::Point2i m_delta_translation;
    // Refinement matched result using template matching
    m_i_kdtree_flann_search_param = 50;
    m_d_radius_of_pixel           = 20;   // 찾을 경계의 반경
    m_i_max_num_near_points       = 50;     // 찾을 주변점의 최대 개수
    m_i_boundary_gap              = m_d_radius_of_pixel * 2;
    m_i_template_size             = 25;
    m_i_num_min_matched_pair      = 5;

    return true;
}

// Match two image using correspondece set via descriotor or other various methods.
// Input  : Two images, keypoints, descriotors or correpondence setes.
// Output : vector of DMatch object and result images.
bool imageMatcher::MatchImages( cv::Mat in_img_1, cv::Mat in_img_2,
                                std::vector<cv::KeyPoint> in_keypoint_1, std::vector<cv::KeyPoint> in_keypoint_2,
                                cv::Mat in_desc_1, cv::Mat in_desc_2,
                                KeyDescType in_desc_type,
                                std::vector<cv::DMatch> &out_vec_dmatch )
{
    if (in_img_1.empty() || in_img_1.empty())
    {
        std::cerr << "[imageMatcher][MatchImage] Image is empty." << std::endl;
        return false;
    }

    int i_num_closest_point;
    if (m_b_do_ratio_test)
        i_num_closest_point = 2;
    else
        i_num_closest_point = 1;
    // Match two set of descriptors
    std::vector<std::vector<cv::DMatch>> vvec_dmatch_knn;
    switch (in_desc_type)
    {
        case TSIFT:
        {
            cv::Ptr<cv::DescriptorMatcher> ptr_descriptor_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            ptr_descriptor_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn, i_num_closest_point);
            break;
        }
        case TSURF:
        {
            cv::Ptr<cv::DescriptorMatcher> ptr_descriptor_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            // SURF is a floating-point descriptor NORM-L2 is used. -OPENCV-
            ptr_descriptor_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn, i_num_closest_point);
            break;
        }
        case TBRIEF:
        {
            cv::Ptr<cv::DescriptorMatcher> ptr_descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); // witch one is better? BruteForce-Hamming or cv::DescriptorMatcher::FLANNBASED?
            ptr_descriptor_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn, i_num_closest_point);
            break;
        }
        case TORB:
        {
            cv::Ptr<cv::DescriptorMatcher> ptr_descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); // witch one is better? BruteForce-Hamming or cv::DescriptorMatcher::FLANNBASED?
            ptr_descriptor_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn, i_num_closest_point);
            /* code */
            break;
        }
        case TAKAZE:
        {
            cv::Ptr<cv::DescriptorMatcher> ptr_descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); // witch one is better? BruteForce-Hamming or cv::DescriptorMatcher::FLANNBASED?
            ptr_descriptor_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn, i_num_closest_point);
            break;
        }
        case TSLIC:
        {
            std::vector<std::vector<cv::DMatch>> vvec_dmatch_knn_one_to_many;
            cv::Ptr<cv::DescriptorMatcher> ptr_sift_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            ptr_sift_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn_one_to_many, m_i_num_one_to_many);
            // Histogram voting using geographical(지리적) infomation.
            cv::Point p_delta_translation;
            GetCenterOfGeographyConstraint(vvec_dmatch_knn_one_to_many, in_keypoint_1, in_keypoint_2, p_delta_translation);
            RefineMatchedResult(in_img_1, in_img_2, in_keypoint_1, in_keypoint_2, p_delta_translation, out_vec_dmatch);
            return true;
        }
        default:
        {
            std::cerr << "[imageMatcher][MatchImage] The Keypoint Type is wrong." << std::endl;
            return false;
        }
    }

    // Filter matches using the Lowe's ratio test.
    if (m_b_do_ratio_test)
    {
        for (size_t idx = 0; idx < vvec_dmatch_knn.size(); idx++)
        {
            if(vvec_dmatch_knn[idx][0].distance < m_f_ratio_th * vvec_dmatch_knn[idx][1].distance)
            {
                out_vec_dmatch.push_back(vvec_dmatch_knn[idx][0]);
            }
        }
    }
    else
    {
        for (size_t idx = 0; idx < vvec_dmatch_knn.size(); idx++)
        {
            out_vec_dmatch.push_back(vvec_dmatch_knn[idx][0]);
        }
    }

    return true;
}


void imageMatcher::ShowMatchingResult( cv::Mat in_img_1, cv::Mat in_img_2, 
                                       std::vector<cv::KeyPoint> in_vec_keypoints_1, std::vector<cv::KeyPoint> in_vec_keypoints_2,
                                       std::vector<cv::DMatch> in_vec_dmatch, cv::Mat &out_img_matched)
{
    cv::drawMatches( in_img_1, in_vec_keypoints_1, in_img_2, in_vec_keypoints_2, in_vec_dmatch, out_img_matched,
                     cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("Matching Result", out_img_matched);
    cv::waitKey();
    cv::destroyAllWindows();
}


cv::Mat imageMatcher::GetHomography( cv::Mat in_img_1, cv::Mat in_img_2,
                                     std::vector<cv::KeyPoint> in_vec_keypoint_1, std::vector<cv::KeyPoint> in_vec_keypoint_2,
                                     std::vector<cv::DMatch> in_vec_dmatch, cv::Mat in_matched_img)
{
    // Localize the object
    std::vector<cv::Point2f>  vec_p2f_object;
    std::vector<cv::Point2f>  vec_p2f_scene;
    for (size_t idx = 0; idx < in_vec_dmatch.size(); idx++)
    {
        vec_p2f_object.push_back( in_vec_keypoint_1[in_vec_dmatch[idx].queryIdx].pt );
        vec_p2f_scene.push_back( in_vec_keypoint_2[in_vec_dmatch[idx].trainIdx].pt );
    }

    // Find Homography.
    cv::Mat mat_mask;
    cv::Mat mat_homography = cv::findHomography(vec_p2f_object, vec_p2f_scene, cv::RANSAC, 0.5,mat_mask);

    // Get the corners from the image_1.
    std::vector<cv::Point2f> vec_p2f_obj_corners(4);
    vec_p2f_obj_corners[0] = cv::Point2f(0,0);
    vec_p2f_obj_corners[1] = cv::Point2f((float)in_img_1.cols,0);
    vec_p2f_obj_corners[2] = cv::Point2f((float)in_img_1.cols,(float)in_img_1.rows);
    vec_p2f_obj_corners[3] = cv::Point2f(0,(float)in_img_1.rows);
    std::vector<cv::Point2f> vec_p2f_scene_corners(4);
    
    // Check if the homography is working okay.
    // find four corner of the map img using homography with RANSAC algorithm.
    cv::perspectiveTransform(vec_p2f_obj_corners, vec_p2f_scene_corners, mat_homography);

    cv::Mat mat_match_result;
    cv::drawMatches(in_img_1, in_vec_keypoint_1, in_img_2, in_vec_keypoint_2, in_vec_dmatch, mat_match_result,
                     cv::Scalar::all(-1), cv::Scalar::all(-1), mat_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Draw Lines for Verification.
    cv::line( mat_match_result, vec_p2f_scene_corners[0] + cv::Point2f((float)in_img_1.cols, 0),
              vec_p2f_scene_corners[1] + cv::Point2f((float)in_img_1.cols, 0), cv::Scalar(0, 0, 255), 8);
    cv::line( mat_match_result, vec_p2f_scene_corners[1] + cv::Point2f((float)in_img_1.cols, 0),
              vec_p2f_scene_corners[2] + cv::Point2f((float)in_img_1.cols, 0), cv::Scalar(0, 0, 255), 8);
    cv::line( mat_match_result, vec_p2f_scene_corners[2] + cv::Point2f((float)in_img_1.cols, 0),
              vec_p2f_scene_corners[3] + cv::Point2f((float)in_img_1.cols, 0), cv::Scalar(0, 0, 255), 8);
    cv::line( mat_match_result, vec_p2f_scene_corners[3] + cv::Point2f((float)in_img_1.cols, 0),
              vec_p2f_scene_corners[0] + cv::Point2f((float)in_img_1.cols, 0), cv::Scalar(0, 0, 255), 8);


    cv::imshow("Good Matches & Localization", mat_match_result);
    cv::waitKey();
    cv::destroyAllWindows();

    return mat_match_result;
}

bool imageMatcher::RefineMatchedResult( cv::Mat in_uav_img, cv::Mat in_map_img, 
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
    std::vector<cv::Point2i> vecp_interest_keypoints_loc;

    for ( int idx_keypoint_map = 0; idx_keypoint_map < in_vec_map_keypoints.size();  idx_keypoint_map++)
    {
        int i_keypoint_x = in_vec_map_keypoints[idx_keypoint_map].pt.x;
        int i_keypoint_y = in_vec_map_keypoints[idx_keypoint_map].pt.y;
        
        if ( i_left_boundary <= i_keypoint_x && i_keypoint_x <= i_right_boundary && i_top_boundary  <= i_keypoint_y && i_keypoint_y <= i_bot_boundary )
        {
            vecp_interest_keypoints_loc.push_back(cv::Point2i(i_keypoint_x, i_keypoint_y));
            std::cout << "interest Keypoints x,y: " << i_keypoint_x << ", " << i_keypoint_y << std::endl;
        }
    } // 여기까지는 전체 맵 키포인트에서 필요한 부분만 일차적으로 추출해서 벡터에 넣어놓는 과정

    if ( vecp_interest_keypoints_loc.empty() )
    {
        std::cerr << "interest boundary is empty." << std::endl;
        return false;
    }

    
    // Search and push back to vector
    for (size_t idx_query = 0; idx_query < in_vec_uav_keypoints.size(); idx_query++) // UAV 이미지에 있는 키포인트만큼 반복
    {
        int i_query_keypoint_x = in_vec_uav_keypoints[idx_query].pt.x;
        int i_query_keypoint_y = in_vec_uav_keypoints[idx_query].pt.y;
        // UAV 키포인트 주변의 Map 키포인트를 걸러내기 위한 바운더리 설정
        int i_boudnary_left    = i_query_keypoint_x - m_d_radius_of_pixel - in_translation.x; 
        int i_boudnary_right   = i_query_keypoint_x + m_d_radius_of_pixel - in_translation.x;
        int i_boudnary_top     = i_query_keypoint_y - m_d_radius_of_pixel - in_translation.y;
        int i_boudnary_bot     = i_query_keypoint_y + m_d_radius_of_pixel - in_translation.y;
        
        // 일차적으로 추출한 Map 키포인트들 중 가까운 애들만 찾는 과정
        for (int idx_near_keypoint = 0; idx_near_keypoint < vecp_interest_keypoints_loc.size(); idx_near_keypoint++) 
        {
            // 일차적으로 뽑아 놓은 Map 키포인트 중 하나의 위치 정보 추출
            int i_near_point_x = vecp_interest_keypoints_loc[idx_near_keypoint].x;
            int i_near_point_y = vecp_interest_keypoints_loc[idx_near_keypoint].y;
            
            // 바운더리 검사해서 바운더리 안쪽에 있는지 검사
            if( i_boudnary_left < i_near_point_x && i_near_point_x < i_boudnary_right && i_boudnary_top < i_near_point_y && i_near_point_y < i_boudnary_bot)
            {
                // 지정된 바운더리 안쪽에 있는 점이라면 탬플릿 매칭
                int i_ncc_score = 0;
                for (int idx_row = -m_i_template_size; idx_row <= m_i_template_size*2; idx_row++)
                {
                    const double* ptr_elem_uav = in_uav_img.ptr<double>( i_query_keypoint_x + idx_row);
                    const double* ptr_elem_map = in_uav_img.ptr<double>( i_near_point_x + idx_row);
                    for(int idx_col = -m_i_template_size; idx_col <= m_i_template_size*2; idx_col++)
                    {
                        i_ncc_score =+         ptr_elem_uav[i_query_keypoint_y + idx_col] * ptr_elem_map[i_near_point_y + idx_col] /
                                        cv::sqrt(ptr_elem_uav[i_query_keypoint_y + idx_col] * ptr_elem_uav[i_query_keypoint_y + idx_col] *
                                                 ptr_elem_map[i_near_point_y + idx_col] * ptr_elem_map[i_near_point_y + idx_col]);
                        // 한 키포인트에 대해 탬플릿 매칭이 끝나면 mapidm_refined_matching 맵에 스코어 값과 매칭 정보를 같이 넣음
                        // Map에 넣으면 자동으로 정렬
                    }
                }
                // 하나의 UAV 키포인트에 대해 검사가 다 끝나면 점수가 가장 높은 점을 Dmatch로 만들어서 저장
            }
        }
        
        // std::cout << "biggest  near key point y: " << i_biggest_y  << std::endl;
        // std::cout << "biggest  near key point x: " << i_biggest_x  << std::endl;
        // std::cout << "smallest near key point x: " << i_smallest_x << std::endl;
        // std::cout << "smallest near key point y: " << i_smallest_y << std::endl;
        
        // if( vecp2i_near_keypoint_loc.empty() == true )
        // {
        //     std::cout << "[RefineMatchedResult] There is no near keypoints." << std::endl;
        //     continue;
        // }

        if ( b_dbg_vis_close_key_point_extract == true )
        {
            cv::Mat mat_interest_keypoints_img;
            cv::Mat mat_query_keypoints_img;
            in_map_img.copyTo(mat_interest_keypoints_img);
            in_uav_img.copyTo(mat_query_keypoints_img);
            // ShowKeypoints(mat_interest_keypoints_img, veckey_near_keypoints, mat_interest_keypoints_img);
            cv::circle(mat_query_keypoints_img, cv::Point(i_query_keypoint_x, i_query_keypoint_y),5,cv::Scalar(0,0,255),1,-1,0);
            cv::imshow("[Refinement Matched Result] Interest Keypoints near query point", mat_interest_keypoints_img);
            cv::imshow("[Refinement Matched Result] Query point", mat_query_keypoints_img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }    
    }

    
    end = std::clock();
    duration = (double)(end-start);
    std::cout << "Template matching based refinement duration: " << duration/CLOCKS_PER_SEC << "s." << std::endl;
    return true;
}

// Find Geographical constraints using one to many constrainst
bool imageMatcher::GetCenterOfGeographyConstraint( std::vector<std::vector<cv::DMatch>> in_vvec_dmatch_reuslt,
                                                    std::vector<cv::KeyPoint> in_vec_uav_keypoints,
                                                    std::vector<cv::KeyPoint> in_vec_map_keypoints,
                                                    cv::Point &out_center_location)
{
    if( in_vvec_dmatch_reuslt.size() < m_i_num_min_matched_pair)
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

void imageMatcher::ShowKeypoints(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_key_points, cv::Mat &out_mat_keypoint_img)
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


bool imageMatcher::SetGeographicConstraints(cv::Point2i in_delta_translation)
{
    m_delta_translation.x = in_delta_translation.x;
    m_delta_translation.y = in_delta_translation.y;
    return true;
}