clear

% need to enter paths and image size below depending on the selected view
%-------------------------
% enter path where cropped mouth ROIs are located
sourcePath = 'OuluVS2\cropped_mouth_mp4_phrase'; 


% select view
subFolder = '1'; % frontal
% subFolder = '2'; % 30%
% subFolder = '3'; %45
% subFolder = '4'; %60
% subFolder = '5'; %profile

% select file to save the dataset (depending on which view is chosen)
file2Write = 'oulu\data\allMouthROIsResized_frontal';

% file2Write = 'OuluVS2\allMouthROIsResized_30';

% file2Write = 'OuluVS2\allMouthROIsResized_45';

% file2Write = 'OuluVS2\allMouthROIsResized_60';

% file2Write = 'OuluVS2\allMouthROIsResized_profile';

% select mouth ROI size depending on the view
targetH = 29; % frontal
targetW = 50; % frontal
% targetH = 29; % 30 degrees
% targetW = 44; % 30 degrees
% targetH = 29; % 45 degrees
% targetW = 43; % 45 degrees
% targetH = 35; % 60 degrees
% targetW = 44; % 60 degrees
% targetH = 44; % profile
% targetW = 30; % profile

%-------------------------

d = dir(sourcePath);

d(1:2) = [];

noFolders = length(d);
ind = 0;

for i = 1:noFolders
   
    folderName = d(i).name;
    newPath = [sourcePath,filesep,folderName,filesep,subFolder];
    
    vidDir = dir([newPath,filesep,'*.mp4']);
    
    for vi = 1:length(vidDir)
        
        ind = ind + 1;
        
        videoName = vidDir(vi).name;
        videoPath = [newPath,filesep,videoName];
        vObj = VideoReader(videoPath);
        v = read(vObj);
        
        [h,w,noChannels, noFrames] = size(v);

        % get the subject ID from the filename
        [tok, rem] = strtok(videoName,'_');
        tok(1) = [];
        subjectsVec(ind) = str2double(tok);
        
        %get target and iteration from filename
        [tok2, rem2] = strtok(rem,'_');
        rem2(1:2) = [];
        rem2(end-3:end) = [];
        trg = computeTargetsPhrases(str2double(rem2));
        iter = computeIter(str2double(rem2));  
        
        targetsPerVideoVec(ind) = trg;
        videoLengthVec(ind) = noFrames;
        filenamesVec{ind} = videoName;
        iterVec(ind) = iter;
        
        targetsVecCell{ind} = trg*ones(noFrames,1);
        
        grayFrameResizedMat = zeros(noFrames, targetH*targetW);
        
        for frameNo = 1:noFrames
            %convert rgb frame to grayscale
            grayFrame = rgb2gray(v(:,:,:,frameNo));

            % resize image
            grayFrameResized = imresize(grayFrame, [targetH targetW]);
            grayFrameResizedMat(frameNo,:) = grayFrameResized(:)';

        end

        dataMatrixCells{ind} = grayFrameResizedMat;
    end
    
end

dataMatrix = cell2mat(dataMatrixCells');
targetsVec = cell2mat(targetsVecCell');

filenamesVec = filenamesVec';
iterVec = iterVec';
subjectsVec = subjectsVec';
targetsPerVideoVec = targetsPerVideoVec';
videoLengthVec=videoLengthVec';
dataMatrixCells = dataMatrixCells';

save(file2Write,'filenamesVec','iterVec', 'subjectsVec', 'targetsPerVideoVec', 'targetsVec', 'videoLengthVec','dataMatrix','dataMatrixCells','targetH','targetW')

