%% Deep Neural Network for Classification

% start with a clean slate
close all, clear all, clc, 

%% Parameters
epochs_auto1 = 100;   %400
epochs_auto2 = 100;   %100 
epochs_softmax = 100; %400
epochs_final = 100;   %400

% hideen layers outputs
outputSize1 = 100; % reduction size (number of neurons in hidden layer)
outputSize2 = 50;% reduction size (number of neurons in hidden layer)
outputSize3 = 10; % number of possible outcomes

imageHeight = 28;
imageWidth = 28;

%% Load Train and Test MNIST Data
% ========================================================================
% load train and test MNIST background data
% from Yann LaCun
%load('./data/mnist_original/mnist_original');

% from Bengio
load('./data/mnist_basic/mnist_basic');
%load('./data/mnist_rotation/mnist_rotation');
%load('./data/mnist_background_images/mnist_background_images');
%load('./data/mnist_background_random/mnist_background_random');
%load('./data/mnist_background_rotation/mnist_background_rotation');

% fix labels
tTrain = fixMNISTlabels(tTrain);
tTest = fixMNISTlabels(tTest);
 
inputSize = size(xTrain,1);

% % for configuration use only first 5000 records
% xTrain = xTrain(:,1:5000);
% xTest = xTest(:,1:5000);
% tTrain = tTrain(:,1:5000);
% tTest = tTest(:,1:5000);
 
% view some images
xTrainImages = matrix2imgvector(xTrain,28,28);
viewImages(xTrainImages,5,5);

%% Training the first Autoencoder
% ========================================================================
autoenc1 = autoencoder(xTrain,outputSize1,epochs_auto1); % create and train autoencoder
%view(autoenc1); % view diagram autoencoder

% Visualizing the Results from the first Autoencoder
W1 = autoenc1.IW{1};
weightsImage = helperWeightsToImageGallery(W1,imageHeight,imageWidth,10,10);
imshow(weightsImage);

% Create first hidden layer uses weights and bias trained in autencoder1
autoencHid1 = autoencoderHid(autoenc1,inputSize,outputSize1);
%view(autoencHid1);

% Generate first features
features1 = autoencHid1(xTrain);

%% Training the second Autoencoder
%  Train autoencoder on the features generated from the prev autoencoder.
% ========================================================================
autoenc2 = autoencoder(features1,outputSize2,epochs_auto2); % create and train autoencoder
%view(autoenc2);

% Create second hidden layer uses weights and bias trained in autencoder2
autoencHid2 = autoencoderHid(autoenc2,outputSize1,outputSize2);
%view(autoencHid2);

% Generate second features
features2 = autoencHid2(features1);

%% Training the final Softmax Layer
% ========================================================================
% create a softmax layer, and train it on the output from the
% hidden layer of the second autoencoder. 
finalSoftmax = softmaxLayer(outputSize2,outputSize3,epochs_softmax);

% train the softmax layer in a supervised fashion
% using labels for the training data.
finalSoftmax = train(finalSoftmax,features2,tTrain);
%view(finalSoftmax);

%% Forming a Multilayer Neural Network
% ========================================================================
%view(autoencHid1);
%view(autoencHid2);
%view(finalSoftmax);

% join layers together to form a multilayer neural network. 
% copy the weights and biases from the autoencoders and softmax layer.
finalNetwork = finalNetworkConfig(autoencHid1,... % hidden layer 1
                                  autoencHid2,... % hidden layer 2
                                  finalSoftmax,...% final layer 
                                  inputSize,...   % input size
                                  outputSize1,... % hidden1 size
                                  outputSize2,... % hidden2 size
                                  outputSize3,... % output size
                                  epochs_final);
%view(finalNetwork);

%% Test Data
% ========================================================================
% With the full deep network, compute the results on the test data. 
%y = finalNetwork(xTest);

% visualize the results with a confusion matrix. 
%plotconfusion(tTest,y);

%% Fine tuning the Deep Neural Network
% ========================================================================
% perform backpropagation on the whole multilayer network. 
% fine tune the network by retraining it on the training data in a
% supervised fashion. 
finalNetwork = train(finalNetwork,xTrain,tTrain);
y = finalNetwork(xTest);
plotconfusion(tTest,y);

