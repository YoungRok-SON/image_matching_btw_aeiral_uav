% Image Correction Script using homography
% This script is for Image Correction using imu that represent orientation
% of camera. Using this information, the image that is captured with
% different angle will be corrected as orthophoto.
clc; clear; close all;
%% Upload Test Image

file_path = "C:\Users\Alien08\aerial_map_based_illegal_detection\aerial_map_generation\aerial_data\Engineering_B\sensor_data"
[logging_data_file_name, logging_data_dir_path] = uigetfile('*.JPG', 'Please select a image file.', file_path);
image = imread( strcat(logging_data_dir_path, logging_data_file_name) );
imshow(image);

%% Transform using Camera Orientation

% From RGB to Gray scale image
gray_image = im2gray(image);

% Image resize to specific size.
target_width = 1920;
target_size = [NaN target_width];
resized_image = imresize(gray_image, target_size);
[h,w,~] = size(resized_image)
imshow(resized_image)

% Image 
tf = affine3d;
roll = 0; pitch = 0; yaw = 30;
camera_orientation = [roll pitch yaw];
tf = eul2tform(deg2rad(camera_orientation));

virtual_square = [1 1 1 1;
                  1 -1 1 1;
                  -1 -1 1 1;
                  -1 1 1 1];
centerized_virtual_square = virtual_square;
tf(3,4) = 1;
centerized_virtual_square(:,3) = 0;
moved_virtual_square = (tf * centerized_virtual_square')';
tf_add_perspective = [1 0 0 0;
                      0 1 0 0;
                      0 0 1 0;
                      0 0 1 0;];
min_height = min(moved_virtual_square(:,3));
if min_height < 0
    moved_virtual_square(:,3) = moved_virtual_square(:,3) -min_height;
end
add_perspective = (tf_add_perspective * moved_virtual_square')';

remove_depth_add_perspective(1,:) = add_perspective(1,:)/abs(add_perspective(1,4));
remove_depth_add_perspective(2,:) = add_perspective(2,:)/abs(add_perspective(2,4));
remove_depth_add_perspective(3,:) = add_perspective(3,:)/abs(add_perspective(3,4));
remove_depth_add_perspective(4,:) = add_perspective(4,:)/abs(add_perspective(4,4));

depth_perspective_added_square = remove_depth_add_perspective;
depth_perspective_added_square(:,3) = add_perspective(:,3);

plot3(virtual_square(:,1), virtual_square(:,2),virtual_square(:,3), ...
        'Marker','.','MarkerSize',10,'Color','r');
hold on
plot3(moved_virtual_square(:,1), moved_virtual_square(:,2),moved_virtual_square(:,3), ...
        'Marker','.','MarkerSize',10,'Color','g');
plot3(remove_depth_add_perspective(:,1), remove_depth_add_perspective(:,2),remove_depth_add_perspective(:,3), ...
        'Marker','.','MarkerSize',10,'Color','b');
plot3(depth_perspective_added_square(:,1), depth_perspective_added_square(:,2),depth_perspective_added_square(:,3), ...
        'Marker','.','MarkerSize',10,'Color','black');

moving_points = [remove_depth_add_perspective(:,1)*size(resized_image,2)+size(resized_image,2)/2 remove_depth_add_perspective(:,2)*size(resized_image,1)/2+size(resized_image,1)/2];
fixed_points = [virtual_square(:, 1)*size(resized_image,2)+size(resized_image,2)/2 virtual_square(:, 2)*size(resized_image,1)+size(resized_image,1)/2];
tform = fitgeotrans(moving_points,fixed_points,'projective');

corrected_square = (inv(tform.T) * remove_depth_add_perspective(:,1:3)')'
plot3(corrected_square(:,1), corrected_square(:,2),corrected_square(:,3), ...
        'Marker','.','MarkerSize',10,'Color','magenta');
non_perspective_image = imwarp(resized_image,tform.invert,'OutputView',imref2d(size(resized_image)));
figure
imshow(non_perspective_image )
figure
imshow(resized_image )
figure
imshowpair(non_perspective_image,resized_image)
imwrite(non_perspective_image, "corrected_img.jpg")