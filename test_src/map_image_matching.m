% This script is for image matching between sensor data(image) and
% map(image). This code is only for feasibility check.
clc; clear; close all;
%% Load Map Image
map_image_file = "full_map_scenario_transparent_mosaic_group1.tif";
map_image = imread(map_image_file);
map_image_rgb = map_image(:,:,1:3);
imshow(map_image_rgb);

%%
%%%%%%%%%%%%%%%%%%%%%%%%% Map Image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get Feature from Map
Num_strong_features = 20000;

% 그레이스케일로 변환 & Resize
target_size = [size(map_image_rgb,1)/2 size(map_image_rgb,2)/2];
map_image_rgb_resized = imresize(map_image_rgb, target_size);
map_gray_image = im2gray(map_image_rgb_resized);

% Get ROI From big map
% left_top = [766 2807]; % xy on image plane: x+ = left -> right, y+: top -> bottom
% right_bottom = [1710 1725]; % 
cropped_image = imcrop(map_gray_image); %, [left_top right_bottom]
imshow(cropped_image);
% 이미지로부터 피쳐를 추출
% Input: gray image
% output: KAZEPoints object with additional options specified by one or more Name,Value pair arguments

map_feature_points = detectSIFTFeatures(cropped_image);
% Location: (x,y) coordinate of features
% Scale: 피쳐로 뽑힌 점의 ROI의 크기(scalar)
% Metric: intensity of feature point. it uses a determinant of a approximated Hessian.
% Orientation: specified as an angle in radians. The angle measured from
% the x-axis with the origin set by location input.

[map_features,map_points] = extractFeatures(cropped_image,map_feature_points);

map_image_strong_feature = selectStrongest(map_feature_points, Num_strong_features);
[map_strong_features,map_points] = extractFeatures(cropped_image,map_image_strong_feature);

% Visulization
figure('Name', 'Map Image')
imshow(cropped_image);
hold on 
plot(selectStrongest(map_feature_points, Num_strong_features));

%%
%%%%%%%%%%%%%%%%%%%%%%%%% Sensor Image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Sensor Image
sensor_image_folder_path = "../aerial_data/Engineering_B/sensor_data/orthophotos_100m/";
sensor_image_file = "DJI_0376.JPG";
% "corrected_img.jpg"
sensor_image = imread(sensor_image_folder_path + sensor_image_file);
imshow(sensor_image);

%% Get Feature from Sensor Image
sensor_gray_image = im2gray(sensor_image);
target_size = [1440 1920];
sensor_gray_image = imresize(sensor_gray_image, target_size);
sensor_feature_points = detectSIFTFeatures(sensor_gray_image);
[sensor_image_features,sensor_image_points] = extractFeatures(sensor_gray_image, sensor_feature_points);

% Visualization
sensor_image_figure = figure('Name','Sensor Image');
imshow(sensor_gray_image);
hold on 
plot(selectStrongest(sensor_feature_points, 5000));

%%
%%%%%%%%%%%%%%%%%%%%%%%%% Matching Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Match Images using Two Set of Features
clear troms
tforms = affine2d(eye(3));
confidenceValue = 99.9;
maxNumTrials = 2000;
% Find correspondence between I(n-1) and I(n).
indexPairs = matchFeatures(sensor_image_features, map_strong_features, Unique=true);
% Get matching points
matchedpoints = sensor_image_points(indexPairs(:,1),:);
map_matchedpoints = map_points(indexPairs(:,2),:);
% Estimate the transformation between I(n) and I(n-1).
[tforms, inliner] = estimateGeometricTransform2D( matchedpoints, map_matchedpoints, ...
                                        "affine", Confidence=confidenceValue, maxNumTrials=maxNumTrials);
inlier_points_sensor = matchedpoints(inliner,:);
inlider_points_map = map_matchedpoints(inliner,:);
figure; 
ax = axes;
showMatchedFeatures(sensor_gray_image, cropped_image, matchedpoints, map_matchedpoints,"montag",Parent=ax);
showMatchedFeatures(sensor_gray_image, cropped_image, inlier_points_sensor, inlider_points_map,"montag",Parent=ax);

title(ax,"Candidate point matches");
legend(ax,"Matched points 1","Matched points 2");
%% Move Images using Transformation

figure('Name', "Matched Result.");
imshow(cropped_image)
hold on

% set rotation reference frame to map image.
output_view = imref2d(size(cropped_image));

% Rotate Sensor Image
rotated_image = imwarp(sensor_gray_image, tforms, 'OutputView',output_view);
rotated_image_rgb = imwarp(sensor_image, tforms, 'OutputView',output_view);

imshowpair(rotated_image, cropped_image)

