/**
 * @copyright (c) AI LAB - Konkuk Uni.
 * <br>All rights reserved. Subject to limited distribution and restricted disclosure only.
 * @author  dudfhe3349@gmail.com
 * @file utility.hpp
 * @brief Detect Keypoint of image using various method.
 * @version 1.0
 * @date 07-03-2023
 * @bug No known bugs
 * @warning No warnings
 */

#ifndef __UTILITY__
#define __UTILITY__

#include <iostream>

enum KeyDescType
{
    TSIFT,
    TSURF,
    TBRIEF,
    TORB,
    TAKAZE,
    TSLIC
};

inline 
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
    if (in_key_desc_type == TSLIC)
        return std::string("SLIC");
    // Just in case we add a new item in the future and forget to update this function
    return std::string("???");
}

inline
bool comp(const std::pair<int, int> &p1,const std::pair<int, int> &p2){
    if(p1.second == p2.second){     //빈도수가 같으면 
        return p1.first < p2.first; //숫자(key)작은게 앞으로 
    }
    return p1.second > p2.second;    //다르면 빈도수가 큰게 앞으로 
}

#endif // __UTILITY__