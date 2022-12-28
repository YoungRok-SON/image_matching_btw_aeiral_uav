# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 22:04:39 2022

@author: Alien08
"""

# Trial for image matching between aerial image with UAV acquried image.

# Parameters initialization
#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils

#SLIC
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util         import img_as_float
from skimage              import io


from exif import Image
#%% About Map image Loading
map_type = "konkuk_part" # 1. full_map, 2.konkuk_full, 3.konkuk_part

aerial_map_path      = "C:/Users/Alien08/aerial_map_based_illegal_detection/02_map_images/";
if map_type == "full_map":
    aerial_map_file_name = "aerial_orthomap_konkuk_25cm.tif";
elif map_type == "konkuk_full":
    aerial_map_file_name = "aerial_map_konkuk_25cm.png";
else: # map_type == "konkuk_part"
    aerial_map_file_name = "aerial_map_student_building_25cm.png";
        
aerial_map_path_name = aerial_map_path + aerial_map_file_name;
# About UAV iamge loading
uav_img_path         = "C:/Users/Alien08/aerial_map_based_illegal_detection/01_uav_images/orthophotos_100m/";
uav_img_file_name    = "DJI_0378.JPG";
uav_img_path_name    = uav_img_path + uav_img_file_name ;

#%% About Resize
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
resize_factor        = gsd_uav_img/gsd_aerial_map; # resize factor to match gsd of two image
target_size_uav_img  = np.int16(np.array([width_image, height_image]) * resize_factor);

#%% About Orientation matching
target_orientation   = 130; # [deg]

#%% About Key point extraction using SLIC

# UAV_image
num_superpixels_uav    = 1000; # Desired number of superpixels
num_iterations_uav     = 5;   # Number of pixel level iterations. The higher, the better quality
prior_uav              = 2;   # For shape smoothing term. must be [0, 5]
num_levels_uav         = 5;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
num_histogram_bins_uav = 2;   # Number of histogram bins


num_superpixels_map    = 1000; # Desired number of superpixels
num_iterations_map     = 5;   # Number of pixel level iterations. The higher, the better quality
prior_map              = 5;   # For shape smoothing term. must be [0, 5]
num_levels_map         = 5;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
num_histogram_bins_map = 3;   # Number of histogram bins

num_slic_pixel_gap     = 5;
#%% Load Aerial Map data
map_img     = cv.imread(aerial_map_path_name,cv.IMREAD_COLOR);
imgplot_map = plt.imshow(map_img);
cv.imshow('Map Image', map_img);
cv.waitKey(0)
cv.destroyAllWindows()
#%% Load UAV Map Data
uav_img     = cv.imread(uav_img_path_name, cv.IMREAD_COLOR);
imgplot_uav = plt.imshow(uav_img);
cv.imshow('UAV Image', uav_img);
cv.waitKey(0)
cv.destroyAllWindows()
#%% Image preprocessing

# Image Resize
downsampled_uav_img = cv.resize(uav_img,target_size_uav_img);
imgplot_uav_downsampled = plt.imshow(downsampled_uav_img);
cv.imshow('Downsampled UAV Image', downsampled_uav_img);
cv.waitKey(0)
cv.destroyAllWindows()
# Orientation Matching: 일단 손으로 대강 맞추고 나중에 드론으로 할 때에는 드론 헤딩이랑 같이 쓰지 뭐..
aligned_uav_img = imutils.rotate_bound(downsampled_uav_img, target_orientation);
imgplot_uav_aligned = plt.imshow(aligned_uav_img);
cv.imshow('Aligned UAV Image', aligned_uav_img);
cv.waitKey(0)
cv.destroyAllWindows()


#%% Image matching using SIFT key points and descriptor

# # Initiate SIFT feature detector object.
# sift_detector = cv.SIFT_create();
# # Find the key points and descriptors with SIFT
# key_points_uav,    descriptor_uav    = sift_detector.detectAndCompute(aligned_uav_img,None);
# key_points_aerial, descriptor_map = sift_detector.detectAndCompute(map_img,None);
# # Initiate brute-force matching object.
# brute_force_matcher = cv.BFMatcher();
# indices_matched     = brute_force_matcher.knnMatch(descriptor_uav, descriptor_map, k=2);
# # Apply ratio test
# good_matches = [];
# for m,n in indices_matched:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append([m]);
# # cv.drawMatchesKnn expects list of lists as matches.
# mated_image = cv.drawMatchesKnn(aligned_uav_img, key_points_uav, map_img, key_points_aerial,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv.imshow('Matched Result',mated_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
#%%  Image Matching using SLIC boundary points and SIFT descriptor using SLIC of OPENCV ---> UAV

# Convert color space
converted_uav_img = cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2HSV);
# Get infomation of converted Image
height_uav, width_uav, channels_uav = converted_uav_img.shape;
# Initialize SEEDS Algorithm
seeds = cv.ximgproc.createSuperpixelSEEDS(width_uav, height_uav, channels_uav, num_superpixels_uav, num_levels_uav, prior_uav, num_histogram_bins_uav);
# Run SEEDS
seeds.iterate(aligned_uav_img, num_iterations_uav);
# Get number of superpixel
num_of_superpixels_result = seeds.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)
# Retrieve the segmentation result
labels = seeds.getLabels() # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position


# draw contour
label_mask = seeds.getLabelContourMask(False)
croped_label_mask = cv.bitwise_and(label_mask,label_mask,mask=cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2GRAY))
tmp_aligned_uav_img = aligned_uav_img.copy();

# Draw color coded image
color_img = np.zeros((height_uav, width_uav, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(np.int8(croped_label_mask))
result_bg = cv.bitwise_and(tmp_aligned_uav_img, tmp_aligned_uav_img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=np.int8(croped_label_mask))
result = cv.add(result_bg, result_fg)
cv.imshow('SLIC Key points of UAV Image', result)
cv.waitKey(0)
cv.destroyAllWindows()

# Conver to Grayscale Image for remove useless SLIC point on image boudary.
aligned_uav_img_gray = cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2GRAY)
# Extract key points from slic boundary points. 
key_point_pixel_list_uav = [];
for row in range(num_slic_pixel_gap, height_uav-num_slic_pixel_gap):
    for col in range(num_slic_pixel_gap, width_uav-num_slic_pixel_gap):
        if (croped_label_mask[row,col]       != 0                   and
            aligned_uav_img_gray[row + num_slic_pixel_gap,col ]   != 0 and 
            aligned_uav_img_gray[row - num_slic_pixel_gap,col ]   != 0 and
            aligned_uav_img_gray[row, col - num_slic_pixel_gap]   != 0 and
            aligned_uav_img_gray[row ,col + num_slic_pixel_gap]   != 0   ):
            key_point_pixel_list_uav.append([row, col]);

# Generate Keypoint object list from pixel value.
key_point_list_uav = [];
for i in range(len(key_point_pixel_list_uav)):
    key_point_list_uav.append(cv.KeyPoint(x=key_point_pixel_list_uav[i][1],y=key_point_pixel_list_uav[i][0], size=1))            
            
# Compute descriptor vector from key points.
sift_detector_uav = cv.SIFT_create();
key_points_uav, descriptor_uav = sift_detector_uav.compute(aligned_uav_img,key_point_list_uav,None)

#%%  Image Matching using SLIC boundary points and SIFT descriptor using SLIC of OPENCV ---> Map

# Convert Image from BGR to RGB
converted_map_img = cv.cvtColor(map_img, cv.COLOR_BGR2HSV);
# Get information of map image
height_map, width_map, channels_map = converted_map_img.shape;

# Initialize SEEDS Algorithm
seeds = cv.ximgproc.createSuperpixelSEEDS(width_map, height_map, channels_map, num_superpixels_map, num_levels_map,  prior_map, num_histogram_bins_map);
# Run SEEDS
seeds.iterate(converted_map_img, num_iterations_map);
# Get number of superpixel
num_of_superpixels_result = seeds.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)
# Retrieve the segmentation result
labels = seeds.getLabels() # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position

# draw contour
label_mask = seeds.getLabelContourMask(False)
croped_label_mask = cv.bitwise_and(label_mask,label_mask,mask=cv.cvtColor(map_img, cv.COLOR_BGR2GRAY))
tmp_map_img = map_img;

# Draw color coded image
color_img = np.zeros((height_map, width_map, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(np.int8(croped_label_mask))
result_bg = cv.bitwise_and(tmp_map_img, tmp_map_img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=np.int8(croped_label_mask))
result = cv.add(result_bg, result_fg)
cv.imshow('SLIC Key points of Map Image', result)
cv.waitKey(0)
cv.destroyAllWindows()

# Conver to Grayscale Image for remove useless SLIC point on image boudary.
map_img_gray = cv.cvtColor(map_img, cv.COLOR_BGR2GRAY)
# Extract key points from slic boundary points. 
key_point_pixel_list_map = [];
for row in range(num_slic_pixel_gap, height_uav-num_slic_pixel_gap):
    for col in range(num_slic_pixel_gap, width_uav-num_slic_pixel_gap):
        if (croped_label_mask[row,col]       != 0 and
            map_img_gray[row + num_slic_pixel_gap,col ]   != 0 and 
            map_img_gray[row - num_slic_pixel_gap,col ]   != 0 and
            map_img_gray[row, col - num_slic_pixel_gap]   != 0 and
            map_img_gray[row ,col + num_slic_pixel_gap]   != 0   ):
            key_point_pixel_list_map.append([row, col]);

# Generate Keypoint object list from pixel value.
key_point_list_map = [];
for i in range(len(key_point_pixel_list_map)):
    key_point_list_map.append(cv.KeyPoint(x=key_point_pixel_list_map[i][1], y=key_point_pixel_list_map[i][0], size=1))            
            
# Compute descriptor vector from key points.
sift_detector_map = cv.SIFT_create();
key_points_map, descriptor_map = sift_detector_map.compute(aligned_uav_img,key_point_list_map,None)

#%% Feature matching using Brute-force 

# Initiate brute-force matching object.
# brute_force_matcher = cv.BFMatcher();
# indices_matched     = brute_force_matcher.knnMatch(descriptor_uav, descriptor_map,k=2);
# # Apply ratio test
# good_matches = [];
# for m,n in indices_matched:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append([m]);
# # cv.drawMatchesKnn expects list of lists as matches.
# for i in range(int((len(good_matches)-1)/20)):
#     mated_image = cv.drawMatchesKnn(aligned_uav_img, key_points_uav, map_img, key_points_aerial,good_matches[i*20:(i+1)*20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     cv.imshow('Matched Result',mated_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
    


#%% Check Specific one feature point among SLIC key points.
idx_key_points = 1
key_point_aligned_uav = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[idx_key_points][1],key_point_pixel_list_uav[idx_key_points][0]), 5, (0,0,255))

cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
cv.waitKey(0)
cv.destroyAllWindows()    

#%% Calculate Euclidean distance between two SLIC key point sets.

queried_descriptor_vector_uav = descriptor_uav[idx_key_points];
distance_btw_vectors = [];
# Calcualte distance btw two descriptor sets.
for i in range(len(descriptor_map)):
    queried_descriptor_vector = descriptor_map[i];
    distance_btw_vectors.append((i,np.linalg.norm(queried_descriptor_vector_uav-queried_descriptor_vector)))
#  sort by distance with index.
distance_btw_vectors.sort(key=lambda x:x[1]);

# Show the best 100 matches
for i in range(100):
    pair_index = distance_btw_vectors[i][0];
    cv.circle()
    

#%% Generate histogram of matched point.
col_pixel_matched_in_aeiral = [];
row_pixel_matched_in_aeiral = [];
for i in range(100):
    col,row = key_point_pixel_list_map[dist[i][0]];
    col_pixel_matched_in_aeiral.append(col);
    row_pixel_matched_in_aeiral.append(row);
plt.hist(row_pixel_matched_in_aeiral)
plt.hist(col_pixel_matched_in_aeiral)

# 그리 그리기
for i in range(100):
    cv.circle(map_img,(row_pixel_matched_in_aeiral[i],col_pixel_matched_in_aeiral[i]), 10, (0,0,255))
cv.imshow("Map features", map_img)
cv.waitKey()
cv.destroyAllWindows()

x,y = key_point_pixel_list_uav[100];
cv.circle(aligned_uav_img,(y, x), 10, (255,0,0))
cv.imshow("UAV features", aligned_uav_img)
cv.waitKey()
cv.destroyAllWindows()
#%% Apply SLIC and extract segmented labels using SLIC of skimage
segmented_label = slic(aligned_uav_img,n_segments=250,compactness=10,sigma=1);

figure_slic = plt.figure("superpixels -- %d segments"%(250));
ax_slic     = figure_slic.add_subplot(1,1,1);
cv.imshow('MaskWindow', mark_boundaries(aligned_uav_img, segmented_label, color=(1,1,1), outline_color=(1,1,1)))
cv.waitKey(0)
cv.destroyAllWindows()

tmp1 = aligned_uav_img;
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(segmented_label)
result_bg = cv.bitwise_and(tmp1, tmp1, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=segmented_label)
result = cv.add(result_bg, result_fg)
cv.imshow('ColorCodedWindow', result)
cv.waitKey(0)
cv.destroyAllWindows()



