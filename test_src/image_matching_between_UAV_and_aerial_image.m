%% Trial for image matching between aerial image with UAV acquried image.
close all; clc; clear;

% Parameters initialization

% About Map image Loading
aerial_map_path = "map_images/";
aerial_map_file_name = "aerial_orthomap_konkuk_25cm.tif";
aerial_map_path_name = aerial_map_path + aerial_map_file_name;

% About UAV iamge loading
uav_image_path = "../aerial_data/sensor_data/orthophotos_100m/";
uav_image_file_name = "DJI_0379.JPG";
uav_image_path_name = uav_image_path + uav_image_file_name ;
%% Load images

% Load Aerial Map data
map_image = imread(aerial_map_path_name);
map_image_rgb = map_image(:,:,1:3);
figure("Name","Map image");
imshow(map_image_rgb);

% Load UAV Map Data
uav_image = imread(uav_image_path_name);
uav_image_rgb = uav_image(:,:,1:3);
figure("Name","UAV image");
imshow(uav_image_rgb);

% Image Resize
focal_length = 24; %mm 35mm 환산 기준
width_lens = 6.4;
altitude_uav = 100;
gsd_uav = 

% Orientation Matching

% Feature point extraction

% Descriptor Extraction

% Image Matching

%% Preprocessing

% To gray scale Image

% Resize Image
%%% UAV Image GSD to Aerial map GSD (?? -> 25cm)
%%% GSD = (CCD * H) / focal length 

% Orientation Matching


%% Image Matching



%% Transform UAV image to aerial image map



%% Visualization of result
