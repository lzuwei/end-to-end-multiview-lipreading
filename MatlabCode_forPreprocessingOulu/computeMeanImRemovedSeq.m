function [meanImRemovedFrames, meanImTable] = computeMeanImRemovedSeq(dataMatrix, lengthVec)

[noEx, dim] = size(dataMatrix);
dataCells = mat2cell(dataMatrix, lengthVec);
noCells = length(dataCells);
meanImTable = zeros(noCells,dim);

clear dataMatrix

for i = 1:noCells
    
    imSeq = dataCells{i};
    noSeq = size(imSeq,1);
    
    meanIm = mean(imSeq);
    imSeq = imSeq - repmat(meanIm, noSeq, 1);

    dataCells{i} = imSeq;
    meanImTable(i,:) = meanIm;

end

meanImRemovedFrames = cell2mat(dataCells);



