% Trying image-map matching using map from 국토지리정보원
clc; clear; close all;
%% Get map from local
map_image_file = "aerial_orthomap_konkuk_25cm.tif";
map_image = imread(map_image_file);

figure('Name','Map Image')
imshow(map_image)

%% Get Image from Drone view

sensor_image_folder_path = "../aerial_data/Engineering_B/sensor_data/";
sensor_image_file = "corrected_img.jpg";
sensor_image = imread(sensor_image_folder_path + sensor_image_file);
imshow(sensor_image);

boundary = [741 2756 2004 4565]; % Left top(x,y) right bottom(x,y)
block_image = blockedImage(map_image);
bigimageshow(block_image);
block_image.Size
hrect = drawrectangle('Position',boundary);
%% Get Feature from Sensor Image
sensor_gray_image = im2gray(sensor_image);
sensor_feature_points = detectKAZEFeatures(sensor_gray_image);
[sensor_image_features,sensor_image_points] = extractFeatures(sensor_gray_image, sensor_feature_points);

% Visualization
sensor_image_figure = figure('Name','Sensor Image');
imshow(sensor_gray_image);
hold on 
plot(selectStrongest(sensor_feature_points, 5000));