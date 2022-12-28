visualization.py

#%% UAV Imag Key point Finding
key_point_aligned_uav = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
for i in range(len(key_point_pixel_list_uav)):
    cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[i][1],key_point_pixel_list_uav[i][0]), 1, (0,0,255),-1)
    cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
    cv.waitKey(0)
    cv.destroyAllWindows()


# %%


idx_key_points = 2000
key_point_aligned_uav = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
cv.circle(key_point_aligned_uav,(key_point_pixel_list_uav[idx_key_points][1],key_point_pixel_list_uav[idx_key_points][0]), 5, (0,0,255))

cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
cv.waitKey(0)
cv.destroyAllWindows()    

#%% Calculate Euclidean distance between two SLIC key point sets.

queried_descriptor_vector_uav = descriptor_uav[idx_key_points];
distance_btw_vectors = [];
# Calcualte distance btw two descriptor sets.
for i in range(len(descriptor_map)):
    queried_descriptor_vector_map = descriptor_map[i];
    distance_btw_vectors.append( (i, np.linalg.norm(queried_descriptor_vector_uav-queried_descriptor_vector_map) ) )
#  sort by distance with index.
distance_btw_vectors.sort(key=lambda x:x[1]);
# Show the best 100 matches
tmp_map_image = map_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
for i in range(num_closest_descriptors):
    pair_index = distance_btw_vectors[i][0];
    cv.circle(tmp_map_image,(key_point_pixel_list_map[pair_index][1],key_point_pixel_list_map[pair_index][0]), 3, (0,255,255),-1)

cv.imshow('Queried Pixel of UAV Image', key_point_aligned_uav);
cv.imshow('Top 100 matches between UAV and Map', tmp_map_image);
cv.waitKey(0)
cv.destroyAllWindows()
