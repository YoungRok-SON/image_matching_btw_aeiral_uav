/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file test_code.hpp
 * @brief All test code for matching two images.
 * @version 1.0
 * @date 03-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __TEST_CODES__
#define __TEST_CODES__

#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"
#include "opencv4/opencv2/xfeatures2d.hpp"

enum KeyDescType
{
    TSIFT,
    TSURF,
    TBRIEF,
    TORB,
    TAKAZE
};

class TestCodes
{
private:
/* Member Variables */

// For SURF Features
int m_i_min_hessian = 400;

// For Ratio test
float m_f_ratio_th  = 0.9;
public:
/* Init Functions */
    TestCodes(/* args */);
    ~TestCodes();
public:
/* Member Functions */
    bool ExtractKeyAndDesc(cv::Mat in_img, KeyDescType in_key_type, KeyDescType in_desc_type, std::vector<cv::KeyPoint> &out_vec_keypoints, cv::Mat &out_mat_descriptors);
    bool DetectKeyPoints (cv::Mat in_img, KeyDescType in_key_type, std::vector<cv::KeyPoint> &out_vec_keypoints);
    bool ComputeDescriptors(cv::Mat in_img, std::vector<cv::KeyPoint> in_veckey_keypoitns, KeyDescType in_desc_type, cv::Mat &out_mat_descriptors);
    bool MatchImages(cv::Mat in_img_1, cv::Mat in_img_2, cv::Mat in_desc_1, cv::Mat in_desc_2, KeyDescType in_desc_type, std::vector<cv::DMatch> &in_vec_dmatch);
    void ShowMatchingResult(cv::Mat in_img_1, cv::Mat in_img_2, 
                            std::vector<cv::KeyPoint> in_vec_keypoints_1, std::vector<cv::KeyPoint> in_vec_keypoints_2,
                            std::vector<cv::DMatch> in_vvec_dmatch, cv::Mat &out_img_matched);
    cv::Mat GetHomography(cv::Mat in_img_1, cv::Mat in_img_2, std::vector<cv::KeyPoint> in_vec_keypoint_1, std::vector<cv::KeyPoint> in_vec_keypoint_2, std::vector<cv::DMatch> in_vec_dmatch, cv::Mat in_matched_img);
private:
/* Variables for Debugging */
    bool m_b_show_match_result;
    bool m_b_do_ratio_test;
    bool m_b_show_keypoint_result;
};





/* ---------------- */
/* Define functions */
/* ---------------- */

TestCodes::TestCodes()
{
    m_b_show_match_result = true;
    m_b_do_ratio_test = true; 
    m_b_show_keypoint_result = false;
}

TestCodes::~TestCodes(){}

// Keypoint Detect and Descriptor Generation
// Input  : Image for matching, key points type and descriptor type.
// Output : vector of keypoints and vector of Descriptor 
bool TestCodes::ExtractKeyAndDesc(cv::Mat in_img, KeyDescType in_key_type, KeyDescType in_desc_type, std::vector<cv::KeyPoint> &out_vec_keypoints, cv::Mat &out_mat_descriptors)
{
    DetectKeyPoints(in_img, in_key_type, out_vec_keypoints);
    ComputeDescriptors(in_img, out_vec_keypoints, in_desc_type, out_mat_descriptors);
    return true;
}

bool TestCodes::DetectKeyPoints(cv::Mat in_img, KeyDescType in_key_type, std::vector<cv::KeyPoint> &out_vec_keypoints)
{

    if (in_img.empty())
    {
        std::cerr << "[TestCodes][DetectKeyPoints]Image is empty." << std::endl;
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
        default:
        {
            std::cerr << "[TestCodes][DetectKeyPoint]The Keypoint Type is wrong." << std::endl;
            return false;
        }
    }
    if (out_vec_keypoints.empty())
    {
        std::cerr << "[TestCodes][DetectKeyPoints] The Keypoint vector is empty." << std::endl;
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

    return true;
}

bool TestCodes::ComputeDescriptors(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_keypoints, KeyDescType in_desc_type, cv::Mat &out_mat_descriptors)
{
    if (in_img.empty())
    {
        std::cerr << "[TestCodes][ComputeDescriptors] Image is empty." << std::endl;
        return false;
    }

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
            cv::Ptr<cv::xfeatures2d::SURF> ptr_surf_detector = cv::xfeatures2d::SURF::create(m_i_min_hessian);
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
        default:
        {
            std::cerr << "[TestCodes][ComputeDescriptors]]The Keypoint Type is wrong." << std::endl;
            return false;
        }
    }

    if (out_mat_descriptors.empty())
    {
        std::cerr << "[TestCodes][DetectKeyPoints] The Keypoint vector is empty." << std::endl;
        return false;
    }

    return true;
}

bool TestCodes::MatchImages(cv::Mat in_img_1, cv::Mat in_img_2, cv::Mat in_desc_1, cv::Mat in_desc_2, KeyDescType in_desc_type, std::vector<cv::DMatch> &out_vec_dmatch)
{
    if (in_img_1.empty() || in_img_1.empty())
    {
        std::cerr << "[TestCodes][MatchImage] Image is empty." << std::endl;
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
        default:
        {
            std::cerr << "[TestCodes][MatchImage] The Keypoint Type is wrong." << std::endl;
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


void TestCodes::ShowMatchingResult(cv::Mat in_img_1, cv::Mat in_img_2, 
                                   std::vector<cv::KeyPoint> in_vec_keypoints_1, std::vector<cv::KeyPoint> in_vec_keypoints_2,
                                   std::vector<cv::DMatch> in_vec_dmatch, cv::Mat &out_img_matched)
{
    cv::drawMatches( in_img_1, in_vec_keypoints_1, in_img_2, in_vec_keypoints_2, in_vec_dmatch, out_img_matched,
                     cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("Matching Result", out_img_matched);
    cv::waitKey();
    cv::destroyAllWindows();
}


cv::Mat TestCodes::GetHomography(cv::Mat in_img_1, cv::Mat in_img_2, std::vector<cv::KeyPoint> in_vec_keypoint_1, std::vector<cv::KeyPoint> in_vec_keypoint_2, std::vector<cv::DMatch> in_vec_dmatch, cv::Mat in_matched_img)
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

std::string getItemName(KeyDescType in_key_desc_type)
{
    if (in_key_desc_type == TSIFT)
        return std::string("SIFT");
    if (in_key_desc_type == TSURF)
        return std::string("SURF");
    if (in_key_desc_type == TORB)
        return std::string("ORB");
    if (in_key_desc_type == TBRIEF)
        return std::string("BRIEF");
    if (in_key_desc_type == TAKAZE)
        return std::string("AKAZE");
    // Just in case we add a new item in the future and forget to update this function
    return std::string("???");
}

#endif //__TEST_CODES__