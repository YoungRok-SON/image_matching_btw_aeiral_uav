# This script is for testing various feature matching method
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 22:04:39 2022

@author: Alien08
"""
# Parameters initialization
#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import time 

#SLIC
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util         import img_as_float
from skimage              import io

from exif import Image
#% About Map image Loading
map_type = "full_map" # 1. full_map, 2.konkuk_full, 3.konkuk_part

aerial_map_path      = "../02_map_images/";
if map_type == "full_map":
    aerial_map_file_name = "geo_tagged_ortho_image_konkuk_latlon.tif";
elif map_type == "konkuk_full":
    aerial_map_file_name = "aerial_map_konkuk_25cm.png";
else: # map_type == "konkuk_part"
    aerial_map_file_name = "aerial_map_student_building_25cm.png";
        
aerial_map_path_name = aerial_map_path + aerial_map_file_name;
# About UAV iamge loading
uav_img_path         = "../01_uav_images/orthophotos_100m/";
uav_img_file_name    = "DJI_0378.JPG";
uav_img_path_name    = uav_img_path + uav_img_file_name ;

#% About Resize
with open(uav_img_path_name, "rb") as f:
    uav_img = Image(f)
# if you wanna see list of meta data then use "dir(uav_img)"

altitude_uav         = uav_img.gps_altitude * 100; # [m to cm]
focal_length         = uav_img.focal_length / 10 ; # [mm to cm]
width_image          = uav_img.image_width;        # [px]
height_image         = uav_img.image_height;       # [px]
width_ccd_sensor     = 6.4/10;                     # [mm to cm] width of ccd sensor: check spec of camera.
gsd_uav_img          = altitude_uav*width_ccd_sensor/(focal_length*width_image); # [cm]
gsd_aerial_map       = 25;                         # [cm] ground sampling distance: Check the information from the institude of aerial image.
gsd_magic_factor     = 1
resize_factor        = gsd_uav_img/gsd_aerial_map*gsd_magic_factor; # resize factor to match gsd of two image
target_size_uav_img  = np.int16(np.array([width_image, height_image]) * resize_factor);

#%% Load Aerial Map data
map_img     = cv.imread(aerial_map_path_name,cv.IMREAD_COLOR);
map_img     = cv.imread(aerial_map_path_name,cv.IMREAD_LOAD_GDAL | cv.IMREAD_COLOR);
# imgplot_map = plt.imshow(map_img);
# cv.imshow('Map Image', map_img);
# cv.waitKey(0)
# cv.destroyAllWindows()
#% Load UAV Map Data
uav_img     = cv.imread(uav_img_path_name, cv.IMREAD_COLOR);
# imgplot_uav = plt.imshow(uav_img);
# cv.imshow('UAV Image', uav_img);
# cv.waitKey(0)
# cv.destroyAllWindows()
#% Image preprocessing
# Image Resize
downsampled_uav_img = cv.resize(uav_img,target_size_uav_img,interpolation=cv.INTER_AREA);
# imgplot_uav_downsampled = plt.imshow(downsampled_uav_img);
# cv.imshow('Downsampled UAV Image', downsampled_uav_img);
# cv.waitKey(0)
# cv.destroyAllWindows()
# Orientation Matching: 일단 손으로 대강 맞추고 나중에 드론으로 할 때에는 드론 헤딩이랑 같이 쓰지 뭐..
# About Orientation matching
target_orientation   = 130; # [deg]
aligned_uav_img = imutils.rotate_bound(downsampled_uav_img, target_orientation);
# imgplot_uav_aligned = plt.imshow(aligned_uav_img);
# cv.imshow('Aligned UAV Image', aligned_uav_img);
# cv.waitKey(0)
# cv.destroyAllWindows()

#%% BRISK keypoint, descriptor based image matching

# Initiate BRISK descriptor
BRISK = cv.BRISK_create()

# Find the keypoints and compute the descriptors for input and training-set image
keypoints_uav, descriptors_uav = BRISK.detectAndCompute(aligned_uav_img, None)
keypoints_map, descriptors_map = BRISK.detectAndCompute(map_img, None)

#% Brute-force based matching
# create BFMatcher object
BFMatcher = cv.BFMatcher(normType = cv.NORM_HAMMING,
                         crossCheck = True)

# Matching descriptor vectors using Brute Force Matcher
matches = BFMatcher.match(queryDescriptors = descriptors_uav,
                          trainDescriptors = descriptors_map)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

for i in range(len(matches)):
    # Draw first 15 matches
    feature_matching_result = cv.drawMatches(img1 = aligned_uav_img,
                                            keypoints1 = keypoints_uav,
                                            img2 = map_img,
                                            keypoints2 = keypoints_map,
                                            matches1to2 = matches[i*10:(i+1)*10],
                                            outImg = None,
                                            flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow('Matched Result', feature_matching_result);
    cv.waitKey(0)
    cv.destroyAllWindows()

# #% FLANN-based feature matching
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params,search_params)

# # Match feature using knnMatch
# num_closest_descriptors = 2;
# matches = flann.knnMatch(descriptors_uav,descriptors_map,k=num_closest_descriptors)

# matches = sorted(matches[:], key = lambda x: x.distance)

# # feature_matching_result = cv.drawMatches(aligned_uav_img, keypoints_uav, map_img, keypoints_map, matches, None, flags=2) 
# feature_matching_result = cv.drawMatchesKnn(aligned_uav_img, keypoints_uav, map_img, keypoints_map, good, None, flags=2) 
# # feature_matching_result = cv.drawMatches(aligned_uav_img, tuple_key_poins_uav, map_img, tuple_key_poins_map, matches, None, flags=2)

# cv.imshow('Matched Result', feature_matching_result);
# cv.waitKey(0)
# cv.destroyAllWindows()


#%% 매칭 포인트가 다 생성되면 이미지 매칭 진행
    
MIN_MATCH_COUNT = 10
if len(good_matches)>MIN_MATCH_COUNT:
    src_pts = np.float32([ key_point_list_uav[dmatch_element.queryIdx].pt for dmatch_element in best_matching ]).reshape(-1,1,2)
    dst_pts = np.float32([ key_point_list_map[dmatch_element.trainIdx].pt for dmatch_element in best_matching ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = aligned_uav_img_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    map_img = cv.polylines(map_img,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
# img3 = cv.drawMatches(aligned_uav_img,key_point_list_uav, map_img,key_point_list_map,best_matching,None,**draw_params)
# plt.imshow(img3, 'gray'),plt.show()
# cv.imshow('Matched Result', img3);
# cv.waitKey(0)
# cv.destroyAllWindows()

#%%
result3 = cv.warpPerspective(aligned_uav_img, M, (width_map,height_map))
alpha = 0.5
beta = 0.1
new_image = np.zeros(map_img.shape,map_img.dtype)
for y in range(map_img.shape[0]):
    for x in range(map_img.shape[1]):
        for c in range(map_img.shape[2]):
            new_image[y,x,c] = np.clip(alpha*map_img[y,x,c] + beta, 0, 255)

overlay = cv.add(new_image,result3)
cv.imshow('result3', overlay)
cv.waitKey(0)
cv.destroyAllWindows()
## To Do 
# 1. histogram voting 잘 되는지 확인 -> Done
# 2. Template matching  버그 수정 -> Done
# 3. uav에 있는 모든 피쳐점에 대해 loop하여 최종 매칭 포인트 찾기 -> Done
# 4. 매칭 정보를 사용하여 homography 찾기  -> Done
# 5. 매칭 결과 보기
