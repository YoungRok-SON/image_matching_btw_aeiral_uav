# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 22:04:39 2022

@author: Alien08
"""

# Trial for image matching between aerial image with UAV acquried image.

# Parameters initialization

% About Map image Loading
aerial_map_path      = "../../02_map_images/";
aerial_map_file_name = "aerial_orthomap_konkuk_25cm.tif";
aerial_map_path_name = aerial_map_path + aerial_map_file_name;

% About UAV iamge loading
uav_img_path         = "../../01_uav_images/orthophotos_100m/";
uav_img_file_name    = "DJI_0378.JPG";
uav_img_path_name    = uav_img_path + uav_img_file_name ;

% About Resize
info_uav_img         = imfinfo(uav_img_path_name);
altitude_uav         = info_uav_img.GPSInfo.GPSAltitude*100;  % [m2cm]
focal_length         = info_uav_img.DigitalCamera.FocalLength/10; % [mm2cm]
width_image          = info_uav_img.Width;  % [px]
height_image         = info_uav_img.Height; % [px]
width_ccd_sensor     = 6.4/10; % width of ccd sensor [mm2cm]
gsd_uav_img          = altitude_uav*width_ccd_sensor/(focal_length*width_image); % [cm]
gsd_aerial_map       = 25; %  [cm]
resize_factor        = gsd_uav_img/gsd_aerial_map;
target_size_uav_img  = int16([height_image width_image]*resize_factor);

% About Orientation matching
target_orientation   = -130; % [deg]

% About Feature point extraction
use_cropped_map           = true; % 안자르고 하려니까 지도 사이즈가 너무 커서 안되겠음 잘라주긴 해야할 듯
method_feature_extraction = 'SLIC'; % SIFT
num_strong_SIFT_feature_point = 5000;
%%% SLIC parameter
num_devided_area_uav    = 50;

% About Feature matching
clear troms
uav_to_map_tform = affine2d(eye(3));
confidenceValue = 99.9;
maxNumTrials = 2000;

%% Load images

% Load Aerial Map data
map_img     = imread(aerial_map_path_name);
map_img_rgb = map_img(:,:,1:3);

% Load UAV Map Data
uav_img     = imread(uav_img_path_name);
uav_img_rgb = uav_img(:,:,1:3);
%% Image preprocessing

% Generate gray scale img
map_img_gray = rgb2gray(map_img);
uav_img_gray = rgb2gray(uav_img);

% Image Resize
downsampled_uav_img = imresize(uav_img_gray,target_size_uav_img);

% Orientation Matching: 일단 손으로 대강 맞추고 나중에 드론으로 할 때에는 드론 헤딩이랑 같이 쓰지 뭐..
rotated_img = imrotate(downsampled_uav_img, target_orientation,"bicubic");

% imshowpair(rotated_img,map_img_rgb,'montage');
figure("Name","UAV image");
imshow(rotated_img);

% Map Image crop
if (use_cropped_map == true)
    map_for_matching = imcrop(map_img_gray);
elseif(use_cropped_map == false)
    map_for_matching = map_img_gray;
end
figure("Name","Map image");
imshow(map_for_matching);
hold on


%% Feature point extraction
if (isequal(method_feature_extraction,"SIFT"))
    %%% Method 1: Using only SIFT feature points
    feature_points_uav    = detectSIFTFeatures(rotated_img);
    feature_points_aerial = detectSIFTFeatures(map_for_matching);
    
    %%%%%% UAV Image
    close all;
    figure("Name","UAV image");
    imshow(rotated_img);
    hold on
    strong_feature_points_uav = feature_points_uav.selectStrongest(num_strong_SIFT_feature_point);
    plot(strong_feature_points_uav,'ShowOrientation',100)
    
    %%%%%% Aerial Image
    figure("Name","Map image");
    imshow(map_for_matching);
    hold on
    plot(feature_points_aerial,'ShowOrientation',10);

elseif(isequal(method_feature_extraction,"SLIC"))
    %%% Calculate Parameter for SLIC of aerial map
    map_size                = size(map_for_matching);
    uav_img_size            = size(rotated_img);
    num_devided_area_aerial = int16(num_devided_area_uav *  (max(map_size)/max(uav_img_size))) * 2; % SLIC 개수... 어떻게 해야할지 3은 매직넘버
    
    %%% Method 2: Using SLIC point as feature points
    [label_uav,    num_of_label_uav]    = superpixels(rotated_img,num_devided_area_uav,'Compactness',10,"Method","slic","NumIterations",100);
    [label_aerial, num_of_label_aeiral] = superpixels(map_for_matching,num_devided_area_aerial,'Compactness',10,"Method","slic","NumIterations",100);
    %%%%%% UAV Image
    figure("Name","UAV SLIC Image");
    boundary_mask_uav = boundarymask(label_uav);
    imshow(imoverlay(rotated_img,boundary_mask_uav,'cyan'), 'InitialMagnification',0.1);
    %%%%%% Aerial Image
    figure("Name","Map SLIC Image")
    boundary_mask_aerial = boundarymask(label_aerial);
    imshow(imoverlay(map_for_matching,boundary_mask_aerial,'cyan'), 'InitialMagnification',0.1);
    num_feature_point = 0;

    % Count number of boundary pixel with value.
    reshaped_boundary_mask_uav = reshape(boundary_mask_uav,[],1);
    reshaped_boundary_mask_aerial = reshape(boundary_mask_aerial,[],1);
    %%% For UAV images
    count_features = 1;
    for num_feature = 1:1:length(reshaped_boundary_mask_uav)
        if(reshaped_boundary_mask_uav(num_feature) == true )
            count_features = count_features + 1;
        end
    end
    slic_points_uav = zeros(count_features, 2);
    %%% For Map images
    count_features = 1;
    for num_feature = 1:1:length(reshaped_boundary_mask_aerial)
        if(reshaped_boundary_mask_aerial(num_feature) == true )
            count_features = count_features + 1;
        end
    end
    slic_points_aerial = zeros(count_features, 2);

    % Extract feature points at the boundary
    cunt_uav = 1;
    for i = 11:1:uav_img_size(1)-11
        for j = 11:1:uav_img_size(2)-11
            if(boundary_mask_uav(i,j) == true && rotated_img(i,j) ~= 0 && rotated_img(i-10,j) ~= 0 && rotated_img(i+10,j) ~= 0 && rotated_img(i,j-10) ~= 0 && rotated_img(i,j+10) ~= 0)
                slic_points_uav(cunt_uav, :) = [j i];
                cunt_uav = cunt_uav+1;
            end
        end
    end
    slic_points_uav(cunt_uav:end,:) = []; % remove 0 pixels

    cunt_aerial = 1;
    for i = 1:1:map_size(1)
        for j = 1:1:map_size(2)
            if(boundary_mask_aerial(i,j) == true && map_for_matching(i,j) ~= 0)
                slic_points_aerial(cunt_aerial, :) = [j i];
                cunt_aerial = cunt_aerial+1;
            end
        end
    end
    slic_points_aerial(cunt_aerial:end,:) = []; % remove 0 pixels

    close all;

    figure("Name","UAV SLIC Image");
    imshow(rotated_img);
    hold on
    plot(slic_points_uav(:,1), slic_points_uav(:,2),'Marker','+','MarkerSize',5,'LineStyle','none','Color','r');

    figure("Name","Map SLIC Image");
    imshow(map_for_matching);
    hold on
    plot(slic_points_aerial(:,1), slic_points_aerial(:,2),'Marker','+','MarkerSize',5,'LineStyle','none','Color','r');
    
    % Descriptor Extraction
    feature_points_uav = SIFTPoints(slic_points_uav);
    feature_points_aerial = SIFTPoints(slic_points_aerial);
end

% Feature extaction
[descriptors_uav, feature_points_uav]       = extractFeatures(rotated_img,feature_points_uav,"Method","SIFT",'Upright',true);
[descriptors_aerial, feature_points_aerial] = extractFeatures(map_for_matching,feature_points_aerial,"Method","SIFT",'Upright',true);
% [descriptors_uav, feature_points_uav] = extractFeatures(rotated_img,feature_points_uav,"Method","SIFT");
% [descriptors_aerial, feature_points_aerial] = extractFeatures(map_for_matching,feature_points_aerial,"Method","SIFT");
% [descriptors_uav, feature_points_uav] = extractFeatures(rotated_img,feature_points_uav,"Method","SURF");
% [descriptors_aerial, feature_points_aerial] = extractFeatures(map_for_matching,feature_points_aerial,"Method","SURF");

%% Image Matching

% Find correspondence between I(n-1) and I(n).
% close all;
figure('Name',"Result of feature point matching"); 
ax = axes;

indices_matched      = matchFeatures( descriptors_uav,descriptors_aerial,Method="Approximate",Unique=false,MatchThreshold=100,MaxRatio=1);
%% Feature Matching using K-means


% Get matching points
matched_points_uav = feature_points_uav(indices_matched(:,1),:);
matched_points_aerial = feature_points_aerial(indices_matched(:,2),:);

for i = 1:1:size(indices_matched,1)
    showMatchedFeatures(rotated_img, map_for_matching, matched_points_uav(1:100), matched_points_aerial(1:100),"montage",Parent=ax);
end

% 
% showMatchedFeatures(rotated_img, map_for_matching, matched_points_uav, matched_points_aerial,"montage",Parent=ax);

%% Estimate the transformation between I(n) and I(n-1).
[uav_to_map_tform, inliner_mached_points] = estimateGeometricTransform2D( matched_points_uav, matched_points_aerial, ...
                                        "affine", Confidence=confidenceValue, maxNumTrials=maxNumTrials);
inlier_points_sensor = matched_points_uav(inliner_mached_points,:);
inlider_points_map = matched_points_aerial(inliner_mached_points,:);
figure('Name',"Result of feature point matching"); 
ax = axes;
showMatchedFeatures(rotated_img, map_for_matching, matched_points_uav, matched_points_aerial,"montag",Parent=ax);
% showMatchedFeatures(rotated_img, map_for_matching, inlier_points_sensor, inlider_points_map,"montag",Parent=ax);

title(ax,"Candidate point matches");
legend(ax,"Matched points 1","Matched points 2");

%% Preprocessing

% To gray scale Image

% Resize Image
%%% UAV Image GSD to Aerial map GSD (?? -> 25cm)
%%% GSD = (CCD * H) / focal length 

% Orientation Matching


%% Image Matching



%% Transform UAV image to aerial image map



%% Visualization of result
