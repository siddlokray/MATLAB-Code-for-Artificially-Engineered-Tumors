% load data
% data split into escaped tumors and non-escaped tumors (binary problem)
imds = imageDatastore('/Users/siddharth/Documents/earlyDetectionArtificialTumor/sorted data', ...
    'IncludeSubfolders',true, ...
    'FileExtensions', '.tif', ...
    'LabelSource', 'foldernames' ... 
    );
imds = shuffle(imds);

% eval_imds = imageDatastore('/Users/siddharth/Documents/earlyDetectionArtificialTumor/evaluation', ...
%     'IncludeSubfolders', true, ...
%     'FileExtensions', '.tif', ...
%     'LabelSource','foldernames' ...
%     );

% split data
% 70-30 split
[train, test] = splitEachLabel(imds, 0.7, 'randomized');
[test, validation] = splitEachLabel(test, 0.667, 'randomized');

% pass through parse function
train.ReadFcn = @parse;
test.ReadFcn = @parse;
validation.ReadFcn = @parse;
% eval_imds.ReadFcn = @parse;

% configure efficientnetb0
% enet = efficientnetb0('Weights', 'imagenet');
% transfer = enet.Layers(2:end-3);
% layers = [
%     imageInputLayer([224,224,3], 'Name', 'input')
%     transfer
%     fullyConnectedLayer(2, 'Name', 'dense')
%     softmaxLayer('Name', 'softmax')
%     classificationLayer('Name', 'classification')
%     ];
% 
% lgraph = layerGraph(layers_1);

% lgraph = @model;

anet = alexnet('Weights', 'imagenet');
transfer = anet.Layers(2:end-3);
layers = [
    imageInputLayer([227,227,3], 'Name', 'input')
    transfer
    fullyConnectedLayer(32, 'Name', 'dense_1', 'WeightL2Factor',0.1)
    dropoutLayer(0.3)
    fullyConnectedLayer(32, 'Name', 'dense_2', 'WeightL2Factor',0.01)
    dropoutLayer(0.3)
    fullyConnectedLayer(2, 'Name', 'dense_3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
    ];

% training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',3e-5, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',1, ...
    'LearnRateDropPeriod',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validation, ...
    'ValidationFrequency',16, ...
    'Verbose',true, ...
    'Plots','training-progress');

% prune
% [nnet,pi,pl,po] = prune(net);

% train network
[net, info] = trainNetwork(train, layers, options);

YPred = classify(net,test);
% YPred = predict(net, test);
accuracy = mean(YPred == test.Labels)
