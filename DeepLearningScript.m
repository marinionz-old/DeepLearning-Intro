%% Explanation
%   We are going to work with CNN (Convolutional Neural Network).
%   We will work with AlexNet: super-processed trained network w/
%   preprocessed images that we are provided.

%   It is based in local analysis (small values "conjuntos" are analyzed
%   each hidden layer).

%   ReLU= transformation of negative values into 0 (of our input values)
%   and then pooling is performed (from local sample, they take the max.
%   values).

%   Machine Learning: characteristics are extracted and analyzed. Set of
%   own features of the images must be provided.

%   Deep Learning: take the image as a WHOLE to be analyzed. Much more
%   useful. LONGER TRAINING TIME, no chosen features needed, LARGER DATASET
%   NEEDED. Few classifiers are available.

%% Exercise I

imds=imageDatastore('RandomImages');
net=alexnet;
% net.Layers in order to see the layers. In the Image input part you can
% see the desired image input (we are using 227 x 227) and put it at the
% same size.

auds=augmentedImageDatastore([227 227],imds);
preds=classify(net, auds);

photo=imread('file01.jpg');
imshow(photo);

%% Exercise II

%   Modifying an already trained network and its layers.

flowerds=imageDatastore('Flowers','IncludeSubfolders',true,'LabelSource','foldernames');

% You can see your Labels' categories, picking 60% (in this case) of the
% images randomly from only files with the demos label.
[train,test]=splitEachLabel(flowerds,0.6);

% How many categories do we have?
NumClass=numel(categories(flowerds.Labels));

net=alexnet;
layers=net.Layers;

% Now we are going to modify two layers, number 23 (number of classes) and 25 (classification)
% Creating a fully connected network layer
layers(end-2)=fullyConnectedLayer(NumClass);
layers(end)=classificationLayer;

% The first option is the mathematical algorithm used to get to our output.
%   SGDM =  gradient descent, slow but precise
%   ADAM =  adaptative moment estimation (conjugate gradient like, that
%   follows the inertia of the convergence, instead of SGDM's orthogonal
%   vectors.
%   InitialRate= speed of the analysis of each layer (k parameter that
%   define the step length). Very big-> very big steps: may not converge.
%   Very small -
options=trainingOptions('sgdm','InitialLearnRate',0.001);

% Train the network
[flowersnet,info] = trainNetwork(trainImgs, layers, options);

%% Exercise 3
% clc
% clear

numclass = 2;
WormData=table2cell(readtable('WormData.csv'));
path0='C:\Users\marti\Desktop\2019-2020\Deep Learning MATLAB\Roundworms\WormImages\';
mkdir('C:\Users\marti\Desktop\2019-2020\Deep Learning MATLAB\Roundworms\WormImages\Alive');
mkdir('C:\Users\marti\Desktop\2019-2020\Deep Learning MATLAB\Roundworms\WormImages\Dead');

for i=1:length(WormData)
    if strcmpi (WormData{i,2},'alive')
        copyfile([path0 WormData{i,1}],[path0 'Alive\']);
    else
        copyfile([path0 WormData{i,1}],[path0 'Dead\']);
    end
end
wormds=imageDatastore('WormImages','IncludeSubfolders',true,'LabelSource','foldernames');
[train,test]=splitEachLabel(wormds,0.6); % The rest 40 % is test.

%% OR 

wormds=imageDatastore('WormImages');
WormData=readtable('WormData.csv');
wormds.Labels=categorical(WormData.Status);


net=alexnet;
[train,test]=splitEachLab
NumClass=2; % alive / deadel(wormds,0.6);
auds=augmentedImageDatastore(

layers=net.Layers;[227 227],wormds);

layers(end-2)=fullyConnectedLayer(NumClass);
layers(end)=classificationLayer;

options=trainingOptions('adam','InitialLearnRate',0.001);

[wormsnet,info] = trainNetwork(train, layers, options);

