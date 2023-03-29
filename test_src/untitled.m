% THis script is about UAV-Aerial image matching using road elements that fly low altitude in urban environment.
clc; clear; close all;

showRawImgs = true;

%% Load Image data
uavImgPath    = "../../01_uav_images/orthophotos_100m/";
mapImgPath    = "../../02_map_images/";
uavSearchTarget = uavImgPath + "*DJI*";


uavImgFileName = dir(uavSearchTarget);
mapImgFileName = mapImgPath + "konkuk_latlon_geo_tagged.tiff";
numUavImg = length(uavImgFileName);
arrUavImg = cell(1, numUavImg);
infoUavImg = cell(1, numUavImg);
fprintf("Loading Images now...\n");
[mapImg, referenceArr] = readgeoraster(mapImgFileName);

for i = 1:1:numUavImg
    imgPathAndName = uavImgPath + uavImgFileName(i).name;
    arrUavImg{i}  = imread(imgPathAndName);
    infoUavImg{i} = imfinfo(imgPathAndName);
end

if(showRawImgs)
    fprintf("Generate Figure for check images...\n");
    rows = 4;
    cols = 3;

    % Show raw drone images
    figure(1);
    montage(arrUavImg, "Size", [rows cols], BorderSize=6);
    
    % Show geo-tagged aerial map
    figure(2);
    geoshow(mapImg, referenceArr, "DisplayType","image");

end

%% Do something fo valueable output
% 1. Seg roof and non-roof
% 2. Extract only ground and ground-attached things.
% 3. Try matching or Extract edges.
% 4. Use NCC or feature matching.

%% Image Seg into Non-roof and roof.

% Get the pretrained network from website.
pretrainedURL = 'https://ssd.mathworks.com/supportfiles/uav/data/deeplabv3plusResnet18Unreal.zip';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetworkZip = fullfile(pretrainedFolder,'deeplabv3plusResnet18Unreal.zip'); 
waitMessage = "Downloading pretrained network (60 MB)...";
exampleHelperDownloadData(pretrainedURL,pretrainedFolder,pretrainedNetworkZip,waitMessage);

% Initialize the colormap, class name, and class index
colorMap     = [60/255  40/255 222/255;
                0       128/255  0  ];
classNames   = ["Non-roof" "Roof"];
classIndices = [0 1];

% Load the pretrained data into RAM.
pretrainedNetwork = fullfile(pretrainedFolder,"pretrainedNetwork.mat");  
load(pretrainedNetwork);

% Classify orthophotos
labelsOrverlaid = {};
for idx = 1:1:numUavImg
    orthoLabels = semanticseg(arrUavImg{idx}, net);
    
    labelsOrverlaid{idx} = labeloverlay(arrUavImg{idx}, orthoLabels, Colormap=colorMap, Transparency=0.4);
end
% Display the predicted ortholabels using a colormap.
figure(3);
montage(labelsOrverlaid, "Size", [rows cols]);
exampleHelperPixelLabelColorbar(colorMap, classNames);
title("Segmented Result");

% Road Segmentation does not work on real world data..
% Need to figure out another way to extract road Things.

%% Get Sub-image from Map.
mapLatLim = referenceArr.LatitudeLimits;
mapLonLim = referenceArr.LongitudeLimits;
mapSubImgCell = cell(1,numUavImg);
[imgWidth, imgHeight, imgChannels] = size(arrUavImg{1});

% Get GSD(Ground Sampling Distance) of UAV Img
ccdWidth  = 4000;


format longEng
for idx = 1:1:numUavImg
    uavLat = infoUavImg{idx}.GPSInfo.GPSLatitude(1) + infoUavImg{idx}.GPSInfo.GPSLatitude(2)/100 + infoUavImg{idx}.GPSInfo.GPSLatitude(3)/10000;
    uavLon = infoUavImg{idx}.GPSInfo.GPSLongitude(1) + infoUavImg{idx}.GPSInfo.GPSLongitude(2)/100 + infoUavImg{idx}.GPSInfo.GPSLongitude(3)/10000;
    uavAlt = infoUavImg{idx}.GPSInfo.GPSAltitude;
    
    altitude_uav         = uav_img.gps_altitude * 100;   % [m]
    focal_length         = uav_img.focal_length / 1000 ; % [mm to m]
    width_image          = uav_img.image_width;          % [px]
    height_image         = uav_img.image_height;         % [px]
    width_ccd_sensor     = 6.4/10;                       % [mm to cm] width of ccd sensor: check spec of camera.
    gsd_uav_img          = altitude_uav*width_ccd_sensor/(focal_length*width_image); % [cm]
    gsd_aerial_map       = 25;                           % [cm] ground sampling distance: Check the information from the institude of aerial image.
    gsd_magic_factor     = 1;
    resize_factor        = gsd_uav_img/gsd_aerial_map*gsd_magic_factor; % resize factor to match gsd of two image
    target_size_uav_img  = np.int16(np.array([width_image, height_image]) * resize_factor); 

    subMapCenterPixel = [  ]
end


%% Image Preprocessing

blurImg     = true;
resizeImg   = true;

bluredImg = cell(1,numUavImg);
blurSigma = 2;

for idx = 1:1:numUavImg
    if blurImg == true
        bluredImg{idx} = imgaussfilt(arrUavImg{idx}, blurSigma);
    end
end

%% Edge extraction using canny edge

uavEdgeImgs = cell(1,numUavImg);
for idx = 1:1:numUavImg
    if blurImg == true
        uavEdgeImgs{idx}   = edge(rgb2gray(bluredImg{idx}),"canny");    
    else 
        uavEdgeImgs{idx}   = edge(rgb2gray(arrUavImg{idx}),"canny");    
    end
end

figure(4);
subplot(1,2,1);
montage(uavEdgeImgs, "Size", [rows cols], ThumbnailSize=[],BorderSize=6, BackgroundColor="White");
title("Extracted Edge");
subplot(1,2,2)
montage(arrUavImg, "Size", [rows cols], ThumbnailSize=[],BorderSize=6, BackgroundColor="White");
title("Raw Image");

%% Save Images whichever  you want.
description = "UAV_Edge_Img";
saveResult = saveProcessedImg(uavEdgeImgs, "C:\Users\Alien08\Desktop\figures\Edge_extraction\", description);

%% Functions for Convenience.
function result = saveProcessedImg(imgCell, saveDir, description)
    fprintf("Images are downloading now...\w")
    for i = 1:1:length(imgCell)
        imwrite(imgCell{i}, saveDir + description + "_" +int2str(i) + ".png");
    end
    fprintf("Downloading Done !\w")
    fprintf(sprintf("Total %d images are saved at ", length(imgCell)) + saveDir + " \w")
    result = true;
end
