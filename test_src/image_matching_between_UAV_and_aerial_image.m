%% Trial for image matching between aerial image with UAV acquried image.
close all; clc; clear;

% Parameters initialization

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

% Orientation matching
target_orientation = -130; % [deg]

% Feature point extraction


%% Load images

% Load Aerial Map data
map_img     = imread(aerial_map_path_name);
map_img_rgb = map_img(:,:,1:3);
figure("Name","Map image");
imshow(map_img_rgb);

% Load UAV Map Data
uav_img     = imread(uav_img_path_name);
uav_img_rgb = uav_img(:,:,1:3);
figure("Name","UAV image");
imshow(uav_img_rgb);

% Generate gray scale img
map_img_gray = rgb2gray(map_img);
uav_img_gray = rgb2gray(uav_img);

% Image Resize
downsampled_uav_img = imresize(uav_img_gray,target_size_uav_img);
figure("Name","UAV image");
imshow(downsampled_uav_img);

% Orientation Matching: 일단 손으로 대강 맞추고 나중에 드론으로 할 때에는 드론 헤딩이랑 같이 쓰지 뭐..
rotated_img = imrotate(downsampled_uav_img, target_orientation, "bilinear");
% imshowpair(rotated_img,map_img_rgb,'montage');
figure("Name","UAV image");
imshow(rotated_img);

% Feature point extraction
%%% Method 1: Using only SIFT feature points


%%% MEthod 2: Using SLIC point as feature points

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
