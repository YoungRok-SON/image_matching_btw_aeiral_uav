map_image_file = "full_map_scenario_transparent_mosaic_group1.tif";
map_image = imread(map_image_file);
map_image_rgb = map_image(:,:,1:3);
imshow(map_image_rgb);

figure('Name','Map Image')
imshow(map_image_rgb)



imshow(KU_orthomosaic)


map_image_file = "full_map_scenario_transparent_mosaic_group1.tif"
map_image = imread(map_image_file);
map_image_rgb = map_image(:,:,1:3);
imshow(map_image_rgb);

figure('Name','Map Image')
imshow(map_image_rgb)