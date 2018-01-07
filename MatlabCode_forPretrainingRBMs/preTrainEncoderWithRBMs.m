
type = 1; % 1 is autoencoder (AE), 2 is classifier 

inputActivationFunction = 'linear'; %sigm for binary inputs, linear for continuous input

% load dataset
load('OuluVS2\allMouthROIsResized_frontal.mat','dataMatrix', 'videoLengthVec', 'subjectsVec')

%-----------------------------
% Oulu
%-----------------------------
% IDs for training/validation/test subjects
trSubj = [1	2 3	5 7 10	11	12	14	16	17	18 19	20	21	23	24	25	27	28	31	32	33 35	36	37	39	40	41 42	45 46	47	48	53];

testSubj = [6	8	9	15	26	30	34	43	44	49	51	52];
 
valSubj = [4 13 22 38 50];

imCells = mat2cell(dataMatrix, videoLengthVec);

% divide data based on training/validation/test IDs
[trainInd,valInd, testInd] = divideDataIntoTrainTestSubjInd(subjectsVec, trSubj, valSubj, testSubj);
trImCells = imCells(trainInd);
trIm = cell2mat(trImCells);
dataMatrix = trIm;

trVideoLengthVec = videoLengthVec(trainInd);
videoLengthVec = trVideoLengthVec;


% compute mean image per sequence
[dataMatrixMeanRemoved, meanImTable] = computeMeanImRemovedSeq(dataMatrix, videoLengthVec);

train_x = dataMatrixMeanRemoved;

inputSize = size(train_x,2);

if type == 1 % AE
   outputSize  = inputSize; % in case of AE it should be equal to the number of inputs

   %if type = 1, i.e., AE then the last layer should be linear and usually a
% series of decreasing layers are used
    hiddenActivationFunctions = {'ReLu','ReLu','ReLu','linear'}; 
    hiddenLayers = [2000 1000 500 50]; 
   
elseif type == 2 % classifier
    outputSize = size(train_y,2); % in case of classification it should be equal to the number of classes

    hiddenActivationFunctions = {'ReLu','ReLu','ReLu'};
    hiddenLayers = [1000 1000 1000 ]; % hidden layers sizes, does not include input or output layers

end

dbnParams = dbnParamsInit(type, hiddenActivationFunctions, hiddenLayers);
dbnParams.inputActivationFunction = inputActivationFunction;
dbnParams.rbmParams.epochs = 20;

% normalise data
train_x = normaliseData(dbnParams.inputActivationFunction, train_x,[]);

% train Deep Belief Network
[dbn, errorPerBatch errorPerSample] = trainDBN(train_x, dbnParams);

nn = unfoldDBNtoNN(dbnParams, dbn, outputSize);

w1 = nn.W{1};
w2 = nn.W{2};
w3 = nn.W{3};
w4 = nn.W{4};
b1 = nn.biases{1};
b2 = nn.biases{2};
b3 = nn.biases{3};
b4 = nn.biases{4};

save(filename,'w1','w2','w3','w4','b1','b2','b3','b4')
