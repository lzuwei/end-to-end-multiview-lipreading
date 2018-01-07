function [trIDs,valIDs, testIDs] = divideDataIntoTrainTestSubjInd(subjIDVec, trSubjIDs, valSubjIDs, testSubjIDs)

trIDcells = {};
valIDcells = {};
testIDcells = {};

for i = 1:length(trSubjIDs)
    
    subjID = trSubjIDs(i);
    tempIDs = find(subjIDVec == subjID);
    
    trIDcells{i} = tempIDs;
    
end


for i = 1:length(valSubjIDs)
    
    subjID = valSubjIDs(i);
    tempIDs = find(subjIDVec == subjID);
    
    valIDcells{i} = tempIDs;
    
end

for i = 1:length(testSubjIDs)
    
    subjID = testSubjIDs(i);
    tempIDs = find(subjIDVec == subjID);
    
    testIDcells{i} = tempIDs;
    
end

trIDs = cell2mat(trIDcells');
valIDs = cell2mat(valIDcells');
testIDs = cell2mat(testIDcells');

