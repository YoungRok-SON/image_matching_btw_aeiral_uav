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

#%% About Feature point extraction
# use_cropped_map           = true; # 안자르고 하려니까 지도 사이즈가 너무 커서 안되겠음 잘라주긴 해야할 듯
# method_feature_extraction = 'SLIC'; # SIFT
# num_strong_SIFT_feature_point = 5000;
# ### SLIC parameter
# num_devided_area_uav    = 50;

# #%% About Feature matching
# clear troms
# uav_to_map_tform = affine2d(eye(3));
# confidenceValue = 99.9;
# maxNumTrials = 2000;

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

# Initiate SIFT feature detector object.
sift_detector = cv.SIFT_create();
# Find the key points and descriptors with SIFT
key_points_uav,    descriptor_uav    = sift_detector.detectAndCompute(aligned_uav_img,None);
key_points_aerial, descriptor_aerial = sift_detector.detectAndCompute(map_img,None);
# Initiate brute-force matching object.
brute_force_matcher = cv.BFMatcher();
indices_matched     = brute_force_matcher.knnMatch(descriptor_uav, descriptor_aerial, k=2);
# Apply ratio test
good_matches = [];
for m,n in indices_matched:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m]);
# cv.drawMatchesKnn expects list of lists as matches.
mated_image = cv.drawMatchesKnn(aligned_uav_img, key_points_uav, map_img, key_points_aerial,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Matched Result',mated_image)
cv.waitKey(0)
cv.destroyAllWindows()
#%%  Image Matching using SLIC boundary points and SIFT descriptor using SLIC of OPENCV ---> UAV

# Convert color space
converted_uav_img = cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2HSV);

num_superpixels    = 1000; # Desired number of superpixels
num_iterations     = 5;   # Number of pixel level iterations. The higher, the better quality
prior              = 2;   # For shape smoothing term. must be [0, 5]
num_levels         = 5;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
num_histogram_bins = 2;   # Number of histogram bins
height, width, channels = aligned_uav_img.shape;

# Initialize SEEDS Algorithm
seeds = cv.ximgproc.createSuperpixelSEEDS(width,height,channels,num_superpixels,num_levels,prior,num_histogram_bins);
# Run SEEDS
seeds.iterate(aligned_uav_img, num_iterations);
# Get number of superpixel
num_of_superpixels_result = seeds.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)
# Retrieve the segmentation result
labels = seeds.getLabels() # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position


# draw contour
label_mask = seeds.getLabelContourMask(False)
croped_label_mask = cv.bitwise_and(label_mask,label_mask,mask=cv.cvtColor(aligned_uav_img, cv.COLOR_BGR2GRAY))
tmp_aligned_uav_img = aligned_uav_img;

# Draw color coded image
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(np.int8(croped_label_mask))
result_bg = cv.bitwise_and(tmp_aligned_uav_img, tmp_aligned_uav_img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=np.int8(croped_label_mask))
result = cv.add(result_bg, result_fg)
cv.imshow('ColorCodedWindow', result)
cv.waitKey(0)
cv.destroyAllWindows()


# Extract key points from slic boundary points. 
key_point_pixel_list_uav = [];
for row in range(height):
    for col in range(width):
        if croped_label_mask[row,col] != 0:
            key_point_pixel_list_uav.append([row, col]);

# Generate Keypoint object list from pixel value.
key_point_list = [];
for i in range(len(key_point_pixel_list_uav)):
    key_point_list.append(cv.KeyPoint(x=key_point_pixel_list_uav[i][0],y=key_point_pixel_list_uav[i][1], size=1))            
            
# Compute descriptor vector from key points.
sift_detector = cv.SIFT_create();
key_points_uav, descriptor_uav = sift_detector.compute(aligned_uav_img,key_point_list,None)

#%%  Image Matching using SLIC boundary points and SIFT descriptor using SLIC of OPENCV ---> Map

converted_map_img = cv.cvtColor(map_img, cv.COLOR_BGR2HSV);

num_superpixels    = 1000; # Desired number of superpixels
num_iterations     = 5;   # Number of pixel level iterations. The higher, the better quality
prior              = 5;   # For shape smoothing term. must be [0, 5]
num_levels         = 5;  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
num_histogram_bins = 3;   # Number of histogram bins
height, width, channels = converted_map_img.shape;

# Initialize SEEDS Algorithm
seeds = cv.ximgproc.createSuperpixelSEEDS(width,height,channels,num_superpixels,num_levels,prior,num_histogram_bins);
# Run SEEDS
seeds.iterate(converted_map_img, num_iterations);
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
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv.bitwise_not(np.int8(croped_label_mask))
result_bg = cv.bitwise_and(tmp_map_img, tmp_map_img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=np.int8(croped_label_mask))
result = cv.add(result_bg, result_fg)
cv.imshow('ColorCodedWindow', result)
cv.waitKey(0)
cv.destroyAllWindows()

# Extract key points from slic boundary points. 
key_point_pixel_list_aerial = [];
for x in range(height):
    for y in range(width):
        if croped_label_mask[x,y] != 0:
            key_point_pixel_list_aerial.append([x,y]);

# Generate Keypoint object list from pixel value.
key_point_list = [];
for i in range(len(key_point_pixel_list_aerial)):
    key_point_list.append(cv.KeyPoint(x=key_point_pixel_list_aerial[i][0],y=key_point_pixel_list_aerial[i][1], size=1))            
            
# Compute descriptor vector from key points.
sift_detector = cv.SIFT_create();
key_points_aerial, descriptor_aerial = sift_detector.compute(map_img,key_point_list,None)

#%% Feature matching using Brute-force 

# Initiate brute-force matching object.
brute_force_matcher = cv.BFMatcher();
indices_matched     = brute_force_matcher.knnMatch(descriptor_uav, descriptor_aerial,k=2);
# Apply ratio test
good_matches = [];
for m,n in indices_matched:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m]);
# cv.drawMatchesKnn expects list of lists as matches.
# for i in range(int((len(good_matches)-1)/20)):
#     mated_image = cv.drawMatchesKnn(aligned_uav_img, key_points_uav, map_img, key_points_aerial,good_matches[i*20:(i+1)*20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     cv.imshow('Matched Result',mated_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
    


#%%

# To do 
# 한 피쳐당 100개까지 매칭 결과 뽑아보기
# 뽑은 결과에 대해서 히스토그램 그려보기
# 몰리는 부분 찾아보기
idx_key_points = 2
key_point_pixel_list_uav[idx_key_points]
key_point_aligned_uav = aligned_uav_img;
cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[idx_key_points][0],key_point_pixel_list_uav[idx_key_points][1]), 10, (0,0,255))

cv.imshow('Aligned UAV Image', key_point_aligned_uav);
cv.waitKey(0)
cv.destroyAllWindows()    

# 유클리디언 
a = descriptor_uav[100];
dist = [];
for i in range(len(descriptor_aerial)):
    b = descriptor_aerial[i];
    dist.append((i,np.linalg.norm(a-b)))


dist.sort(key=lambda x:x[1]);


#%% Generate histogram of matched point.
col_pixel_matched_in_aeiral = [];
row_pixel_matched_in_aeiral = [];
for i in range(100):
    col,row = key_point_pixel_list_aerial[dist[i][0]];
    col_pixel_matched_in_aeiral.append(col);
    row_pixel_matched_in_aeiral.append(row);
plt.hist(x_pixel_matched_in_aeiral)
plt.hist(y_pixel_matched_in_aeiral)

# 그리 그리기
for i in range(100):
    cv.circle(map_img,(y_pixel_matched_in_aeiral[i],x_pixel_matched_in_aeiral[i]), 10, (0,0,255))
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


#%%
elseif(isequal(method_feature_extraction,"SLIC"))
    ### Calculate Parameter for SLIC of aerial map
    map_size                = size(map_for_matching);
    uav_img_size            = size(aligned_uav_img);
    num_devided_area_aerial = int16(num_devided_area_uav *  (max(map_size)/max(uav_img_size))) * 2; # SLIC 개수... 어떻게 해야할지 3은 매직넘버
    
    ### Method 2: Using SLIC point as feature points
    [label_uav,    num_of_label_uav]    = superpixels(aligned_uav_img,num_devided_area_uav,'Compactness',10,"Method","slic","NumIterations",100);
    [label_aerial, num_of_label_aeiral] = superpixels(map_for_matching,num_devided_area_aerial,'Compactness',10,"Method","slic","NumIterations",100);
    ###### UAV Image
    figure("Name","UAV SLIC Image");
    boundary_mask_uav = boundarymask(label_uav);
    imshow(imoverlay(aligned_uav_img,boundary_mask_uav,'cyan'), 'InitialMagnification',0.1);
    ###### Aerial Image
    figure("Name","Map SLIC Image")
    boundary_mask_aerial = boundarymask(label_aerial);
    imshow(imoverlay(map_for_matching,boundary_mask_aerial,'cyan'), 'InitialMagnification',0.1);
    num_feature_point = 0;

    # Count number of boundary pixel with value.
    reshaped_boundary_mask_uav = reshape(boundary_mask_uav,[],1);
    reshaped_boundary_mask_aerial = reshape(boundary_mask_aerial,[],1);
    ### For UAV images
    count_features = 1;
    for num_feature = 1:1:length(reshaped_boundary_mask_uav)
        if(reshaped_boundary_mask_uav(num_feature) == true )
            count_features = count_features + 1;
        end
    end
    slic_points_uav = zeros(count_features, 2);
    ### For Map images
    count_features = 1;
    for num_feature = 1:1:length(reshaped_boundary_mask_aerial)
        if(reshaped_boundary_mask_aerial(num_feature) == true )
            count_features = count_features + 1;
        end
    end
    slic_points_aerial = zeros(count_features, 2);

    # Extract feature points at the boundary
    cunt_uav = 1;
    for i = 11:1:uav_img_size(1)-11
        for j = 11:1:uav_img_size(2)-11
            if(boundary_mask_uav(i,j) == true && aligned_uav_img(i,j) ~= 0 && aligned_uav_img(i-10,j) ~= 0 && aligned_uav_img(i+10,j) ~= 0 && aligned_uav_img(i,j-10) ~= 0 && aligned_uav_img(i,j+10) ~= 0)
                slic_points_uav(cunt_uav, :) = [j i];
                cunt_uav = cunt_uav+1;
            end
        end
    end
    slic_points_uav(cunt_uav:end,:) = []; # remove 0 pixels

    cunt_aerial = 1;
    for i = 1:1:map_size(1)
        for j = 1:1:map_size(2)
            if(boundary_mask_aerial(i,j) == true && map_for_matching(i,j) ~= 0)
                slic_points_aerial(cunt_aerial, :) = [j i];
                cunt_aerial = cunt_aerial+1;
            end
        end
    end
    slic_points_aerial(cunt_aerial:end,:) = []; # remove 0 pixels

    close all;

    figure("Name","UAV SLIC Image");
    imshow(aligned_uav_img);
    hold on
    plot(slic_points_uav(:,1), slic_points_uav(:,2),'Marker','+','MarkerSize',5,'LineStyle','none','Color','r');

    figure("Name","Map SLIC Image");
    imshow(map_for_matching);
    hold on
    plot(slic_points_aerial(:,1), slic_points_aerial(:,2),'Marker','+','MarkerSize',5,'LineStyle','none','Color','r');
    
    # Descriptor Extraction
    feature_points_uav = SIFTPoints(slic_points_uav);
    feature_points_aerial = SIFTPoints(slic_points_aerial);
end

#%% Feature extaction
[descriptors_uav, feature_points_uav]       = extractFeatures(aligned_uav_img,feature_points_uav,"Method","SIFT",'Upright',true);
[descriptors_aerial, feature_points_aerial] = extractFeatures(map_for_matching,feature_points_aerial,"Method","SIFT",'Upright',true);
# [descriptors_uav, feature_points_uav] = extractFeatures(aligned_uav_img,feature_points_uav,"Method","SIFT");
# [descriptors_aerial, feature_points_aerial] = extractFeatures(map_for_matching,feature_points_aerial,"Method","SIFT");
# [descriptors_uav, feature_points_uav] = extractFeatures(aligned_uav_img,feature_points_uav,"Method","SURF");
# [descriptors_aerial, feature_points_aerial] = extractFeatures(map_for_matching,feature_points_aerial,"Method","SURF");

## Image Matching

# Find correspondence between I(n-1) and I(n).
# close all;
figure('Name',"Result of feature point matching"); 
ax = axes;

indices_matched      = matchFeatures( descriptors_uav,descriptors_aerial,Method="Approximate",Unique=false,MatchThreshold=100,MaxRatio=1);
## Feature Matching using K-means


# Get matching points
matched_points_uav = feature_points_uav(indices_matched(:,1),:);
matched_points_aerial = feature_points_aerial(indices_matched(:,2),:);

for i = 1:1:size(indices_matched,1)
    showMatchedFeatures(aligned_uav_img, map_for_matching, matched_points_uav(1:100), matched_points_aerial(1:100),"montage",Parent=ax);
end

# 
# showMatchedFeatures(aligned_uav_img, map_for_matching, matched_points_uav, matched_points_aerial,"montage",Parent=ax);

## Estimate the transformation between I(n) and I(n-1).
[uav_to_map_tform, inliner_mached_points] = estimateGeometricTransform2D( matched_points_uav, matched_points_aerial, ...
                                        "affine", Confidence=confidenceValue, maxNumTrials=maxNumTrials);
inlier_points_sensor = matched_points_uav(inliner_mached_points,:);
inlider_points_map = matched_points_aerial(inliner_mached_points,:);
figure('Name',"Result of feature point matching"); 
ax = axes;
showMatchedFeatures(aligned_uav_img, map_for_matching, matched_points_uav, matched_points_aerial,"montag",Parent=ax);
# showMatchedFeatures(aligned_uav_img, map_for_matching, inlier_points_sensor, inlider_points_map,"montag",Parent=ax);

title(ax,"Candidate point matches");
legend(ax,"Matched points 1","Matched points 2");
