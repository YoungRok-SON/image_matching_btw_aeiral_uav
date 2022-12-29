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
gsd_magic_factor     = 1
resize_factor        = gsd_uav_img/gsd_aerial_map*gsd_magic_factor; # resize factor to match gsd of two image
target_size_uav_img  = np.int16(np.array([width_image, height_image]) * resize_factor);

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
downsampled_uav_img = cv.resize(uav_img,target_size_uav_img,interpolation=cv.INTER_AREA);
imgplot_uav_downsampled = plt.imshow(downsampled_uav_img);
cv.imshow('Downsampled UAV Image', downsampled_uav_img);
cv.waitKey(0)
cv.destroyAllWindows()
# Orientation Matching: 일단 손으로 대강 맞추고 나중에 드론으로 할 때에는 드론 헤딩이랑 같이 쓰지 뭐..
# About Orientation matching
target_orientation   = 130; # [deg]
aligned_uav_img = imutils.rotate_bound(downsampled_uav_img, target_orientation);
imgplot_uav_aligned = plt.imshow(aligned_uav_img);
cv.imshow('Aligned UAV Image', aligned_uav_img);
cv.waitKey(0)
cv.destroyAllWindows()

# Image Crop for feature matching debuging
# cropped_aligned_uav_img = aligned_uav_img[300:700, 300:700];
# cropped_aligned_uav_img = aligned_uav_img[250:500, 200:500]; # without magic factor
# cropped_map_img         = map_img[250:550, 200:550];
# cropped_aligned_uav_img = aligned_uav_img[180:700, 200:700]; # resize factor 1.9

cv.imshow('Cropped Map Image', map_img);
cv.imshow('Cropped Aligned UAV Image', aligned_uav_img);
cv.waitKey(0)
cv.destroyAllWindows()
#%% Apply Crop
map_img         = cropped_map_img;
aligned_uav_img = cropped_aligned_uav_img;
cv.imshow('Cropped Map Image', cropped_map_img);
cv.imshow('Cropped Aligned UAV Image', cropped_aligned_uav_img);
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

#%% About Key point extraction using SLIC - UAV

num_superpixels_uav    = 500; # Desired number of superpixels
num_iterations_uav     = 20;   # Number of pixel level iterations. The higher, the better quality
prior_uav              = 1;   # For shape smoothing term. must be [0, 5]
num_levels_uav         = 3;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
num_histogram_bins_uav = 10;   # Number of histogram bins

num_slic_pixel_gap_uav     = 1;

# Image Matching using SLIC boundary points and SIFT descriptor using SLIC of OPENCV ---> UAV

# Convert color space
converted_uav_img = cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2HSV);
# Get infomation of converted Image
height_uav, width_uav,channels_uav = converted_uav_img.shape;
# Initialize SEEDS Algorithm
seeds = cv.ximgproc.createSuperpixelSEEDS(width_uav, height_uav, channels_uav, num_superpixels_uav, num_levels_uav, prior_uav, num_histogram_bins_uav);
# Run SEEDS
seeds.iterate(converted_uav_img, num_iterations_uav);
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
for row in range(num_slic_pixel_gap_uav, height_uav-num_slic_pixel_gap_uav):
    for col in range(num_slic_pixel_gap_uav, width_uav-num_slic_pixel_gap_uav):
        if (croped_label_mask[row,col]       != 0                   and
            aligned_uav_img_gray[row + num_slic_pixel_gap_uav,col ]   != 0 and 
            aligned_uav_img_gray[row - num_slic_pixel_gap_uav,col ]   != 0 and
            aligned_uav_img_gray[row, col - num_slic_pixel_gap_uav]   != 0 and
            aligned_uav_img_gray[row ,col + num_slic_pixel_gap_uav]   != 0   ):
            key_point_pixel_list_uav.append([row, col]);

# Generate Keypoint object list from pixel value.
key_point_list_uav = [];
for i in range(len(key_point_pixel_list_uav)):
    key_point_list_uav.append(cv.KeyPoint(x=key_point_pixel_list_uav[i][1],y=key_point_pixel_list_uav[i][0], size=1,angle=0))            
            
# Compute descriptor vector from key points.
sift_detector_uav = cv.SIFT_create();
key_points_uav, descriptor_uav = sift_detector_uav.compute(aligned_uav_img,key_point_list_uav,None)
#%% About Key point extraction using SLIC -Map
num_superpixels_map        = 500; # Desired number of superpixels
num_iterations_map         = 20;   # Number of pixel level iterations. The higher, the better quality
prior_map                  = 1;   # For shape smoothing term. must be [0, 5]
num_levels_map             = 3;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
num_histogram_bins_map     = 10;   # Number of histogram bins
num_slic_pixel_gap_map     = 1;

# Image Matching using SLIC boundary points and SIFT descriptor using SLIC of OPENCV ---> Map

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
for row in range(num_slic_pixel_gap_map, height_map-num_slic_pixel_gap_map):
    for col in range(num_slic_pixel_gap_map, width_map-num_slic_pixel_gap_map):
        if (croped_label_mask[row,col]                        != 0 and
            map_img_gray[row + num_slic_pixel_gap_map,col ]   != 0 and 
            map_img_gray[row - num_slic_pixel_gap_map,col ]   != 0 and
            map_img_gray[row, col - num_slic_pixel_gap_map]   != 0 and
            map_img_gray[row ,col + num_slic_pixel_gap_map]   != 0   ):
            key_point_pixel_list_map.append([row, col]);

# Generate Keypoint object list from pixel value.
key_point_list_map = [];
for i in range(len(key_point_pixel_list_map)):
    key_point_list_map.append(cv.KeyPoint(x=key_point_pixel_list_map[i][1], y=key_point_pixel_list_map[i][0], size=1,angle=0))            
            
# Compute descriptor vector from key points.
sift_detector_map = cv.SIFT_create();
key_points_map, descriptor_map = sift_detector_map.compute(map_img,key_point_list_map,None)

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
list_delta_pixel_x = [0 for i in range(len(descriptor_uav)*100)];
list_delta_pixel_y = [0 for i in range(len(descriptor_uav)*100)];
num_closest_descriptors = 100;
num_pixel_in_bin = 1
number_of_bin_y = int(height_map/num_pixel_in_bin)
number_of_bin_x = int(width_map/num_pixel_in_bin)
pixel_boundary_radius = 50; # find near pixels within this boudnary.

# for iter_uav in range(len(descriptor_uav)):
#     print(iter_uav)
#     idx_key_points = iter_uav
#     # key_point_aligned_uav = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
#     # cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[idx_key_points][1],key_point_pixel_list_uav[idx_key_points][0]), 5, (0,0,255))

#     # Calculate Euclidean distance between two SLIC key point sets.

#     queried_descriptor_vector_uav = descriptor_uav[idx_key_points];
#     distance_btw_vectors = [[0,0] for i in range(len(descriptor_map))];
#     # Calcualte distance btw two descriptor sets.
#     for i in range(len(descriptor_map)):
#         queried_descriptor_vector_map = descriptor_map[i];
#         # distance_btw_vectors.append( (i, np.linalg.norm(queried_descriptor_vector_uav-queried_descriptor_vector_map) ) )
#         distance_btw_vectors = [i, np.sum(np.power(queried_descriptor_vector_uav-queried_descriptor_vector_map,2))]
#     #  sort by distance with index.
#     distance_btw_vectors.sort(key=lambda x:x[1]);
#     # Show the best 100 matches
#     # tmp_map_image = map_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
#     # matching_candidates_by_distance  = [];
#     # r = 255; b = 0; g = 0;
#     x_key_point_from_uav = key_point_pixel_list_uav[iter_uav][1];
#     y_key_point_from_uav = key_point_pixel_list_uav[iter_uav][0];
#     for i in range(num_closest_descriptors):
#         pair_index = distance_btw_vectors[i][0];
#         # g = g + 255/num_closest_descriptors;
#         # cv.circle(tmp_map_image,(key_point_pixel_list_map[pair_index][1], key_point_pixel_list_map[pair_index][0]), 5, (b,g,r))
#         x_key_point_from_map = key_point_pixel_list_map[pair_index][1];
#         y_key_point_from_map = key_point_pixel_list_map[pair_index][0];
#         # matching_candidates_by_distance.append( [x_key_point_from_map,y_key_point_from_map] );
#         x_delta_pixel_distance = x_key_point_from_uav - x_key_point_from_map;
#         y_delta_pixel_distance = y_key_point_from_uav - y_key_point_from_map;
        
#         # list_delta_pixel_x.append([x_delta_pixel_distance, y_delta_pixel_distance]);
#         list_delta_pixel_x[iter_uav*num_closest_descriptors + i ] = x_delta_pixel_distance;
#         list_delta_pixel_x[iter_uav*num_closest_descriptors + i ] = y_delta_pixel_distance;
#%% FLANN-based feature matching

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptor_uav,descriptor_map,k=100)
#%% histogram voting
list_delta_pixel_x = [];
list_delta_pixel_y = [];
number_of_bin_x = 1000
number_of_bin_y = 1000

for query_idx in range(len(descriptor_uav)):
    x_query = key_point_list_uav[matches[query_idx][0].queryIdx].pt[0];
    y_query = key_point_list_uav[matches[query_idx][0].queryIdx].pt[1];
    for train_idx in range(num_closest_descriptors):
        x_train = key_point_list_map[matches[query_idx][train_idx].trainIdx].pt[0];
        y_train = key_point_list_map[matches[query_idx][train_idx].trainIdx].pt[1];
        list_delta_pixel_x.append(x_query - x_train)
        list_delta_pixel_y.append(y_query - y_train)
    
histogram_figure, histograme_axes = plt.subplots(2,1, constrained_layout = True)
histograme_axes[0].grid(True)
histograme_axes[0].set_title('X-direction Pixel Histogram')
histograme_axes[0].set_xlabel('Pixel (30px/bin)')
histograme_axes[0].set_ylabel('number of Pixels of bin')
number_per_bin_x, edge_of_bins_x, _  = histograme_axes[0].hist(list_delta_pixel_x,bins=number_of_bin_x) # returns: n: array or list of arrays, bins: array, patches: BarContainer or list of a single polygon or list of such objects

histogram_figure.suptitle('Matched point X-Y Pixel Histogram', fontsize=16)
histograme_axes[1].grid(True)
histograme_axes[1].set_title('Y-direction Pixel Histogram')
histograme_axes[1].set_xlabel('Pixel (30px/bin)')
histograme_axes[1].set_ylabel('number of Pixels of bin')
number_per_bin_y, edge_of_bins_y, _ = histograme_axes[1].hist(list_delta_pixel_y,bins=number_of_bin_y)
#%% not use
# cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
# cv.imshow('Top distance-based matches between UAV and Map', tmp_map_image);
# cv.waitKey(0)
# cv.destroyAllWindows()    

# #% Generate histogram of matched point.
# col_pixel_matched_in_map = [];
# row_pixel_matched_in_map = [];
# for i in range(num_closest_descriptors):
#     row,col = key_point_pixel_list_map[distance_btw_vectors[i][0]];
#     col_pixel_matched_in_map.append(col);
#     row_pixel_matched_in_map.append(row);
# #% Visualization of Histogram
# histogram_figure, histograme_axes = plt.subplots(2,1, constrained_layout = True)
# histograme_axes[0].grid(True)
# histogram_figure.suptitle('Matched point X-Y Pixel Histogram', fontsize=16)
# histograme_axes[0].set_title('Y-direction Pixel Histogram')
# histograme_axes[0].set_xlabel('Pixel (30px/bin)')
# histograme_axes[0].set_ylabel('number of Pixels of bin')
# number_per_bin_y, edge_of_bins_y, _ = histograme_axes[0].hist(row_pixel_matched_in_map,bins=number_of_bin_y)

# histograme_axes[1].grid(True)
# histograme_axes[1].set_title('X-direction Pixel Histogram')
# histograme_axes[1].set_xlabel('Pixel (30px/bin)')
# histograme_axes[1].set_ylabel('number of Pixels of bin')
# number_per_bin_x, edge_of_bins_x, _  = histograme_axes[1].hist(col_pixel_matched_in_map,bins=number_of_bin_x) # returns: n: array or list of arrays, bins: array, patches: BarContainer or list of a single polygon or list of such objects

# histogram_figure.suptitle('Matched point by Descriptor Vector Distance', fontsize=16)
# image_figure, image_axes = plt.subplots(1,2, constrained_layout = True)
# image_axes[0].imshow(key_point_aligned_uav)
# image_axes[1].imshow(tmp_map_image)

#%% Geometric Match Verification with Histogram Voting.

# Get X-axis histogram max
max_edge_of_bin_y = np.where(number_per_bin_y == number_per_bin_y.max())
print ('maxbin', edge_of_bins_y[max_edge_of_bin_y][0])

# Get Y-axis histogram max
max_edge_of_bin_x = np.where(number_per_bin_x == number_per_bin_x.max())
print ('maxbin', edge_of_bins_x[max_edge_of_bin_x][0])

# Find pixel value where value of both histograms are max.
max_pixel_value_y = int(edge_of_bins_y[max_edge_of_bin_y][0])
max_pixel_value_x = int(edge_of_bins_x[max_edge_of_bin_x][0])
#%% Find matched point using Tx and Ty
key_point_idx = 8000;
radius = 20;
x_low   = key_point_pixel_list_uav[key_point_idx][1] - max_pixel_value_x - radius;
x_upper = key_point_pixel_list_uav[key_point_idx][1] - max_pixel_value_x + radius;
y_low   = key_point_pixel_list_uav[key_point_idx][0] - max_pixel_value_y - radius;
y_upper = key_point_pixel_list_uav[key_point_idx][0] - max_pixel_value_y + radius;

matching_candidate_pixels = []; # pixel(x,y)...

for i in range(len(key_point_list_map)):
    x_key_point_map = key_point_pixel_list_map[i][1];
    y_key_point_map = key_point_pixel_list_map[i][0];
    
    if (x_key_point_map > x_low   and
        x_key_point_map < x_upper and
        y_key_point_map > y_low   and
        y_key_point_map < y_upper ):
        matching_candidate_pixels.append([x_key_point_map, y_key_point_map])

  
# Matching Candidate Visualization
matching_candidate_map_image = map_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
cv.circle(matching_candidate_map_image,(max_pixel_value_x,max_pixel_value_y),pixel_boundary_radius, (255,0,255))

key_point_aligned_uav = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[key_point_idx][1], key_point_pixel_list_uav[key_point_idx][0]), 5, (0,0,255))

for i in range(len(matching_candidate_pixels)):
    cv.circle(matching_candidate_map_image,(matching_candidate_pixels[i][0], matching_candidate_pixels[i][1]), 5, (0,255,255))
cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
cv.imshow('Matching Candidate from Map', matching_candidate_map_image);
cv.waitKey(0)
cv.destroyAllWindows()    

#%% 
# Find near pixel set near pixel value where value of both histograms are max. 
matching_candidate_pixels = []; # pixel(x,y)...
for i in range(num_closest_descriptors):
    x = matching_candidates_by_distance[i][0];
    y = matching_candidates_by_distance[i][1];
    if (x > max_pixel_value_x - pixel_boundary_radius and
        x < max_pixel_value_x + pixel_boundary_radius and
        y > max_pixel_value_y - pixel_boundary_radius and
        y < max_pixel_value_y + pixel_boundary_radius ):
        matching_candidate_pixels.append([x,y])
    
# Matching Candidate Visualization
matching_candidate_map_image = map_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
cv.circle(matching_candidate_map_image,(max_pixel_value_x,max_pixel_value_y),pixel_boundary_radius, (255,0,255))

for i in range(len(matching_candidate_pixels)):
    cv.circle(matching_candidate_map_image,(matching_candidate_pixels[i][0], matching_candidate_pixels[i][1]), 5, (0,255,255))
cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
cv.imshow('Matching Candidate from Map', matching_candidate_map_image);
cv.waitKey(0)
cv.destroyAllWindows()    

histogram_figure.suptitle('Near Matched Pixels', fontsize=16)
image_figure, image_axes = plt.subplots(1,2, constrained_layout = True)
image_axes[0].imshow(key_point_aligned_uav)
image_axes[1].imshow(matching_candidate_map_image)

#%% Find best maching pixel by template matching


#%% 위에 템플릿 매칭까지 다 되면 for문 돌려서 UAV 이미지에서 뽑은 모든 피쳐에 대해 진행하는 코드 작성

#%% 매칭 포인트가 다 생성되면 이미지 매칭 진행