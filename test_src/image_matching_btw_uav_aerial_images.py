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
import time 

#SLIC
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util         import img_as_float
from skimage              import io

from exif import Image
#% About Map image Loading
map_type = "konkuk_full" # 1. full_map, 2.konkuk_full, 3.konkuk_part

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
uav_img_file_name    = "DJI_0379.JPG";
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
cv.imshow('Map Image', map_img);
aligned_uav_img = imutils.rotate_bound(downsampled_uav_img, target_orientation);
imgplot_uav_aligned = plt.imshow(aligned_uav_img);
cv.imshow('Aligned UAV Image', aligned_uav_img);
cv.waitKey(0)
cv.destroyAllWindows()

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
label_mask_uav = seeds.getLabelContourMask(False)
croped_label_mask_uav = cv.bitwise_and(label_mask_uav,label_mask_uav,mask=cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2GRAY))
tmp_aligned_uav_img = aligned_uav_img.copy();

# Draw color coded image
color_img = np.zeros((height_uav, width_uav, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(np.int8(croped_label_mask_uav))
result_bg = cv.bitwise_and(tmp_aligned_uav_img, tmp_aligned_uav_img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=np.int8(croped_label_mask_uav))
result_uav = cv.add(result_bg, result_fg)
cv.imshow('SLIC Key points of UAV Image', result_uav)
cv.waitKey(0)
cv.destroyAllWindows()

# Conver to Grayscale Image for remove useless SLIC point on image boudary.
aligned_uav_img_gray = cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2GRAY)
# Extract key points from slic boundary points. 
key_point_pixel_list_uav = [];
count = 0;
for row in range(num_slic_pixel_gap_uav, height_uav-num_slic_pixel_gap_uav):
    for col in range(num_slic_pixel_gap_uav, width_uav-num_slic_pixel_gap_uav):
        if (croped_label_mask_uav[row,col]       != 0                   and
            aligned_uav_img_gray[row + num_slic_pixel_gap_uav,col ]   != 0 and 
            aligned_uav_img_gray[row - num_slic_pixel_gap_uav,col ]   != 0 and
            aligned_uav_img_gray[row, col - num_slic_pixel_gap_uav]   != 0 and
            aligned_uav_img_gray[row ,col + num_slic_pixel_gap_uav]   != 0 ): #%% Key point sampling
            key_point_pixel_list_uav.append([row, col]);
        count = count + 1;

#% Generate Keypoint object list from pixel value.
key_point_list_uav = [];
for i in range(len(key_point_pixel_list_uav)):
    key_point_list_uav.append(cv.KeyPoint(x=key_point_pixel_list_uav[i][1],y=key_point_pixel_list_uav[i][0], size=10,angle=0))            
            
# Compute descriptor vector from key points.
sift_detector_uav = cv.SIFT_create();
key_points_uav, descriptor_uav = sift_detector_uav.compute(aligned_uav_img,key_point_list_uav,None)
# brisk_detector_uav = cv.BRISK_create();
# key_points_uav, descriptor_uav = brisk_detector_uav.compute(aligned_uav_img,key_point_list_uav,None)

sampled_key_points = aligned_uav_img.copy()
for i in range(len(key_point_list_uav)):
    cv.circle(sampled_key_points,[key_point_pixel_list_uav[i][1],key_point_pixel_list_uav[i][0]], 3, [0,0,255],-1)
cv.imshow('SLIC Key points of UAV Image', sampled_key_points)
cv.waitKey(0)
cv.destroyAllWindows()

#%% About Key point extraction using SLIC -Map
num_superpixels_map        = 2000; # Desired number of superpixels
num_iterations_map         = 20;   # Number of pixel level iterations. The higher, the better quality
prior_map                  = 1;   # For shape smoothing term. must be [0, 5]
num_levels_map             = 4;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
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
label_mask_map = seeds.getLabelContourMask(False)
croped_label_mask_map = cv.bitwise_and(label_mask_map,label_mask_map,mask=cv.cvtColor(map_img, cv.COLOR_BGR2GRAY))
tmp_map_img = map_img;

# Draw color coded image
color_img = np.zeros((height_map, width_map, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(np.int8(croped_label_mask_map))
result_bg = cv.bitwise_and(tmp_map_img, tmp_map_img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=np.int8(croped_label_mask_map))
result_map = cv.add(result_bg, result_fg)


# Conver to Grayscale Image for remove useless SLIC point on image boudary.
map_img_gray = cv.cvtColor(map_img, cv.COLOR_BGR2GRAY)
# Extract key points from slic boundary points. 
key_point_pixel_list_map = [];
for row in range(num_slic_pixel_gap_map, height_map-num_slic_pixel_gap_map):
    for col in range(num_slic_pixel_gap_map, width_map-num_slic_pixel_gap_map):
        if (croped_label_mask_map[row,col]                        != 0 and
            map_img_gray[row + num_slic_pixel_gap_map,col ]   != 0 and 
            map_img_gray[row - num_slic_pixel_gap_map,col ]   != 0 and
            map_img_gray[row, col - num_slic_pixel_gap_map]   != 0 and
            map_img_gray[row ,col + num_slic_pixel_gap_map]   != 0   ):
            key_point_pixel_list_map.append([row, col]);

# Generate Keypoint object list from pixel value.
key_point_list_map = [];
for i in range(len(key_point_pixel_list_map)): # 시각화용 match object 생성해야함
    key_point_list_map.append(cv.KeyPoint(x=key_point_pixel_list_map[i][1], y=key_point_pixel_list_map[i][0], size=10,angle=0))            
            
# Compute descriptor vector from key points.
sift_detector_map = cv.SIFT_create();
key_points_map, descriptor_map = sift_detector_map.compute(map_img,key_point_list_map,None)
# brisk_detector_map = cv.BRISK_create();
# key_points_map, descriptor_map = brisk_detector_map.compute(map_img,key_point_list_map,None)


cv.imshow('SLIC Key points of Map Image', result_map)
cv.waitKey(0)
cv.destroyAllWindows()
#%% Check Specific one feature point among SLIC key points.
list_delta_pixel_x = [0 for i in range(len(descriptor_uav)*100)];
list_delta_pixel_y = [0 for i in range(len(descriptor_uav)*100)];
num_closest_descriptors = 100;
pixel_boundary_radius = 70; # find near pixels within this boudnary.

#% FLANN-based feature matching: Binary-based Descriptor
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=100)   # or pass empty dictionary

# SIFT based method: Vector-based Descriptor
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptor_uav,descriptor_map,k=num_closest_descriptors)
#%% histogram voting
list_delta_pixel_x = [];
list_delta_pixel_y = [];
uav_max_y, uav_max_x, channel = aligned_uav_img.shape
map_max_y, map_max_x, channel = map_img.shape
min_x = 0 - map_max_x
max_x = uav_max_x - 0
min_y = 0 - map_max_y
max_y = uav_max_y - 0

number_of_bin_x = max_x - min_x -1#width_map
number_of_bin_y = max_y - min_y -1#height_map # 이 값에 따라 중심점의 히스토그램 뽑히는게 달라져서 T_x, T_y 구하는게 달라지니 맵 이미지 크기와 같게 하면 될 것 같음.

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

#%% Geometric Match Verification with Histogram Voting.

# Get X-axis histogram max
max_edge_of_bin_x = np.where(number_per_bin_x == number_per_bin_x.max())
print ('maxbin X:', edge_of_bins_x[max_edge_of_bin_x][0])
# Get Y-axis histogram max
max_edge_of_bin_y = np.where(number_per_bin_y == number_per_bin_y.max())
print ('maxbin Y:', edge_of_bins_y[max_edge_of_bin_y][0])


# Find pixel value where value of both histograms are max.
max_pixel_value_y = int(edge_of_bins_y[max_edge_of_bin_y][0])
max_pixel_value_x = int(edge_of_bins_x[max_edge_of_bin_x][0])

#% Find matched point using Tx and Ty

best_matching = [];
matching_difference_threshold = 85000;
duration_feature_matching_filtering = 0;
duration_template_matching = 0;
#%%
# for key_point_idx in range(len(matches)):
# for key_point_idx in range(1000,2000):
key_point_idx = 10000;
# start =  time.time()
radius = 50;
x_query_pixel_uav = key_point_list_uav[matches[key_point_idx][0].queryIdx].pt[0];
y_query_pixel_uav = key_point_list_uav[matches[key_point_idx][0].queryIdx].pt[1];
x_low   = x_query_pixel_uav - max_pixel_value_x - radius;
x_upper = x_query_pixel_uav - max_pixel_value_x + radius;
y_low   = y_query_pixel_uav - max_pixel_value_y - radius;
y_upper = y_query_pixel_uav - max_pixel_value_y + radius;

matching_candidate_pixels = []; # pixel(x,y)...

# Map에 있는 애들 중에서 uav key point 주변 애들을 불러옴
for i in range(len(key_point_list_map)):
    x_key_point_map = key_point_pixel_list_map[i][1];
    y_key_point_map = key_point_pixel_list_map[i][0];
    
    if (x_key_point_map > x_low   and
        x_key_point_map < x_upper and
        y_key_point_map > y_low   and
        y_key_point_map < y_upper    ):
        matching_candidate_pixels.append([x_key_point_map, y_key_point_map, i]) # key_point_pixel_list_map(key_point_list_map)의 해당 인덱스를 같이 가져감
# end = time.time()
# duration_feature_matching_filtering = duration_feature_matching_filtering + end - start


# Matching Candidate Visualization
matching_candidate_map_image = map_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
key_point_aligned_uav = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[key_point_idx][1],key_point_pixel_list_uav[key_point_idx][0]), 5, (0,0,255))

for i in range(len(matching_candidate_pixels)):
    cv.circle(matching_candidate_map_image,(matching_candidate_pixels[i][0], matching_candidate_pixels[i][1]), 5, (0,0,255))
cv.circle(matching_candidate_map_image,(int(x_query_pixel_uav)-max_pixel_value_x,int(y_query_pixel_uav)-max_pixel_value_y),10, (255,0,0),-1)
matching_candidate_map_image = cv.resize(matching_candidate_map_image,[int(matching_candidate_map_image.shape[1]/2), int(matching_candidate_map_image.shape[0]/2)] ,interpolation=cv.INTER_AREA);
cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
cv.imshow('Matching Candidate from Map', matching_candidate_map_image);
cv.waitKey(0)
cv.destroyAllWindows()    
#%%

    
    start = time.time()
    #% Find best maching pixel by template matching
    template_size     = 30;
    x_low             = int(x_query_pixel_uav - template_size);
    x_upper           = int(x_query_pixel_uav + template_size);
    y_low             = int(y_query_pixel_uav - template_size);
    y_upper           = int(y_query_pixel_uav + template_size);


    if( x_low   <  0            or
        x_upper >=  width_uav    or
        y_low   <  0            or
        y_upper >=  height_uav      ): # 템플릿 사이즈만큼의 이미지가 만들어지지 않는 가장자리 부분은 스킵
        is_refinement_done = False;
        continue;
    elif( aligned_uav_img_gray[y_low, x_low]     == 0 or 
          aligned_uav_img_gray[y_low, x_upper]   == 0 or 
          aligned_uav_img_gray[y_upper, x_low]   == 0 or 
          aligned_uav_img_gray[y_upper, x_upper] == 0 ): # 템플릿 사이즈만큼의 이미지가 만들어지지 않는 가장자리 부분은 스킵
        is_refinement_done = False;
        continue;
    else:
        template_img      = aligned_uav_img_gray[ y_low:y_upper, x_low:x_upper].copy();
        matching_difference_idx = [];
        for idx in range(len(matching_candidate_pixels)):
            # Get center pixel from candidate pixel list.
            x_candidate = matching_candidate_pixels[idx][0];
            y_candidate = matching_candidate_pixels[idx][1];
            idx_map_key_point_idx = matching_candidate_pixels[idx][2];
            # Get image from map_img that has same size with template image.
            x_low_map    = x_candidate - template_size;
            x_upper_map  = x_candidate + template_size;
            y_low_map    = y_candidate - template_size;
            y_upper_map  = y_candidate + template_size
            if( x_low_map   <  0            or
                x_upper_map >  width_map    or
                y_low_map   <  0            or
                y_upper_map >  height_map      ): # 템플릿 사이즈만큼의 이미지가 만들어지지 않는 가장자리 부분은 스킵
                is_refinement_done = False;
                break;
            
            base_img = map_img_gray[ y_low_map:y_upper_map, x_low_map:x_upper_map];
            score    = np.sum(np.abs(np.subtract(template_img,base_img, dtype=np.float64)))
            matching_difference_idx.append([idx, score, idx_map_key_point_idx]) # 여기에 map keypoint에 대한 인덱싱 추가하고
            is_refinement_done = True;

        if(is_refinement_done == True):
            matching_difference_idx.sort(key=lambda x:x[1])
            matched_idx = matching_difference_idx[0][0] # 여기서 그걸 받고
            matching_difference = matching_difference_idx[0][1]
            idx_train_idx = matching_difference_idx[0][2]
            if ( matching_difference < matching_difference_threshold ): # Matching score 기준으로 이상한 애들은 필터링 해야함
                x_low_map    = matching_candidate_pixels[matched_idx][0] - template_size; # 여기서 받은걸로 불러와야할 것 같은데
                x_upper_map  = matching_candidate_pixels[matched_idx][0] + template_size;
                y_low_map    = matching_candidate_pixels[matched_idx][1] - template_size;
                y_upper_map  = matching_candidate_pixels[matched_idx][1] + template_size;
                best_matching.append(  cv.DMatch(key_point_idx,idx_train_idx,score) ) 
    end = time.time()
    duration_template_matching = duration_template_matching + end - start
                
print("Duration for matching filtering: ",  duration_feature_matching_filtering//60,"m ",duration_feature_matching_filtering%60,"s")
print("Duration for matching refinement: ",  duration_template_matching//60,"m ",duration_template_matching%60,"s")
best_matching_map = map_img.copy();
cv.circle(best_matching_map,(matching_candidate_pixels[matched_idx][0], matching_candidate_pixels[matched_idx][1]), 3, (0,0,255),-1)
cv.imshow('Queried Pixel of UAV Image', aligned_uav_img);
cv.imshow('Matching Candidate from Map', best_matching_map);
cv.waitKey(0)
cv.destroyAllWindows()
#%% Matching result Visualization
good_matches = tuple(best_matching)
tuple_key_poins_uav = tuple(key_point_list_uav)
tuple_key_poins_map = tuple(key_point_list_map)
feature_matching_result = cv.drawMatches(aligned_uav_img, tuple_key_poins_uav, map_img, tuple_key_poins_map, best_matching[:], None, flags=2)
# feature_matching_result = cv.drawMatches(aligned_uav_img, tuple_key_poins_uav, map_img, tuple_key_poins_map, matches, None, flags=2)
cv.imshow('Matched Result', feature_matching_result);
cv.waitKey(0)
cv.destroyAllWindows()


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
    result_map = map_img.copy()
    result_map = cv.polylines(result_map,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None



#%%
result3 = cv.warpPerspective(aligned_uav_img, M, (width_map,height_map))
alpha = 0.5
beta = 0.1
new_image = np.zeros(result_map.shape,map_img.dtype)
for y in range(result_map.shape[0]):
    for x in range(result_map.shape[1]):
        for c in range(result_map.shape[2]):
            new_image[y,x,c] = np.clip(alpha*result_map[y,x,c] + beta, 0, 255)

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
