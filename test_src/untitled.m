% THis script is about UAV-Aerial image matching using road elements that fly low altitude in urban environment.
clc; clear; close all;

showRawImgs = true;

%% Load Image data
uavImgPath    = "../../01_uav_images/orthophotos_100m/";
mapImgPath    = "../../02_map_images/";
uavSearchTarget = uavImgPath + "*DJI*";


uavImgFileName = dir(uavSearchTarget);
mapImgFileName = mapImgPath + "konkuk_latlon_geo_tagged.tif";
numUavImg = length(uavImgFileName);
arrUavImg = cell(1, numUavImg);
infoUavImg = cell(1, numUavImg);
fprintf("Loading Images now...\n");
[mapImg, referenceArr] = readgeoraster(mapImgFileName, "CoordinateSystemType","geographic");

for i = 1:1:numUavImg
    imgPathAndName = uavImgPath + uavImgFileName(i).name;
    arrUavImg{i}  = imread(imgPathAndName);
    infoUavImg{i} = imfinfo(imgPathAndName);
end

if(showRawImgs)
    fprintf("Generate Figure for check images...\n");
    rows = 2;
    cols = ceil(numUavImg/2);
    figure(1);
    montage(arrUavImg, "Size", [rows cols]);
    
    figure(2);
    R = georefcells(referenceArr.LatitudeLimits, referenceArr.LongitudeLimits,referenceArr.CellExtentInLatitude, referenceArr.CellExtentInLongitude)
    mapshow(mapImg, R)

end


%% Do something fo valueable output



