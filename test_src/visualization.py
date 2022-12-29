#%% 맵 이미지 슬릭 셀 바운더리 시각화
map_slic = map_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
for i in range(len(key_point_pixel_list_map)):
    x = key_point_pixel_list_map[i][1];
    y = key_point_pixel_list_map[i][0];
    cv.circle(map_slic,(x, y), 1, (0,255,255))


cv.imshow('Top 100 matches between UAV and Map', map_slic);
cv.waitKey(0)
cv.destroyAllWindows()    
#%% UAV 이미지 슬릭 셀 바운더리 시각화
uav_slic = aligned_uav_img.copy(); # Copy 해 주어야 서클이 계속 갱신됨
for i in range(len(key_point_pixel_list_uav)):
    x = key_point_pixel_list_uav[i][1];
    y = key_point_pixel_list_uav[i][0];
    cv.circle(uav_slic,(x, y), 1, (0,255,255))


cv.imshow('Top 100 matches between UAV and Map', uav_slic);
cv.waitKey(0)
cv.destroyAllWindows()    