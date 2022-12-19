% Get Geodatic info from made map.
clc; clear; close all;
%% Load Map Image
map_image_file = "map_engineering_bldg_B.tif";
map_image = imread(map_image_file);
map_image_rgb = map_image(:,:,1:3);
imshow(map_image_rgb);

% Get info from image