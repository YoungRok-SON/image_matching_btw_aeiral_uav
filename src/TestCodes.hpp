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
    THOG,
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
float m_f_ratio_th  = 0.7;
public:
/* Init Functions */
    TestCodes(/* args */);
    ~TestCodes();
public:
/* Member Functions */
    bool ExtractKeyAndDesc(cv::Mat in_img, KeyDescType in_key_type, KeyDescType in_desc_type, std::vector<cv::KeyPoint> &out_vec_keypoints, cv::Mat mat_descriptors);
    bool DetectKeyPoints (cv::Mat in_img, KeyDescType in_key_type, std::vector<cv::KeyPoint> &out_vec_keypoints);
    bool ComputeDescriptors(cv::Mat in_img, std::vector<cv::KeyPoint> in_veckey_keypoitns, KeyDescType in_desc_type, cv::Mat mat_descriptors);
    bool MatchImages(cv::Mat in_img_1, cv::Mat in_img_2, cv::Mat in_desc_1, cv::Mat in_desc_2, KeyDescType in_desc_type, std::vector<cv::DMatch> &in_vec_dmatch);
    void ShowMatchingResult(cv::Mat in_img_1, cv::Mat in_img_2, 
                            std::vector<cv::KeyPoint> in_vec_keypoints_1, std::vector<cv::KeyPoint> in_vec_keypoints_2,
                            std::vector<cv::DMatch> in_vvec_dmatch);
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
    m_b_do_ratio_test = false; 
    m_b_show_keypoint_result = true;
}

TestCodes::~TestCodes(){}

// Keypoint Detect and Descriptor Generation
// Input  : Image for matching, key points type and descriptor type.
// Output : vector of keypoints and vector of Descriptor 
bool TestCodes::ExtractKeyAndDesc(cv::Mat in_img, KeyDescType in_key_type, KeyDescType in_desc_type, std::vector<cv::KeyPoint> &out_vec_keypoints, cv::Mat out_mat_descriptors)
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

bool TestCodes::ComputeDescriptors(cv::Mat in_img, std::vector<cv::KeyPoint> in_vec_keypoints, KeyDescType in_desc_type, cv::Mat out_mat_descriptors)
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
            std::cerr << "The Keypoint Type is wrong." << std::endl;
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
            break;
        }
        case TAKAZE:
        {
            cv::Ptr<cv::DescriptorMatcher> ptr_descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); // witch one is better? BruteForce-Hamming or cv::DescriptorMatcher::FLANNBASED?
            ptr_descriptor_matcher->knnMatch(in_desc_1, in_desc_2, vvec_dmatch_knn, i_num_closest_point);
        }
        default:
        {
            std::cerr << "The Keypoint Type is wrong." << std::endl;
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
    return true;
}


void TestCodes::ShowMatchingResult(cv::Mat in_img_1, cv::Mat in_img_2, 
                                   std::vector<cv::KeyPoint> in_vec_keypoints_1, std::vector<cv::KeyPoint> in_vec_keypoints_2,
                                   std::vector<cv::DMatch> in_vec_dmatch)
{
    cv::Mat mat_matching_result;
    cv::drawMatches( in_img_1, in_vec_keypoints_1, in_img_2, in_vec_keypoints_2, in_vec_dmatch, mat_matching_result,
                     cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("Matching Result", mat_matching_result);
    cv::waitKey();
    cv::destroyAllWindows();
}
#endif //__TEST_CODES__