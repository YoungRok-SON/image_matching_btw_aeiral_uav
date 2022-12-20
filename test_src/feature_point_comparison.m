% Feature point comparison
figure (1)
Boundary_mask_uav = boundarymask(label_uav);
imshow(imoverlay(rotated_img,Boundary_mask_uav,'r'), 'InitialMagnification',10);
hold on
plot(feature_points_uav.Location(:,1),feature_points_uav.Location(:,2)...
            ,'Marker','.','MarkerSize',30,'Color','r','LineStyle','none')



figure (2)
Boundary_mask_aerial = boundarymask(label_aerial);
imshow(imoverlay(cropped_map,Boundary_mask_aerial ,'cyan'), 'InitialMagnification',10);
hold on
plot(feature_points_aerial.Location(:,1),feature_points_aerial.Location(:,2)...
            ,'Marker','.','MarkerSize',30,'Color','r','LineStyle','none')
