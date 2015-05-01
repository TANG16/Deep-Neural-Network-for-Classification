%% function finalSoftmax = softmaxLayer(inputSize,outputSize)
% desc: second autoencoder
% inputs: 
% output: 
% 
% =====================================================
% 
function finalSoftmax = softmaxLayer(inputSize,outputSize,epochs)

    % Create an empty network
    finalSoftmax = network;

    % Set the number of inputs and layers
    finalSoftmax.numInputs = 1;
    finalSoftmax.numLayers = 1;

    % Connect the 1st (and only) layer to the 1st input, and connect the 1st
    % layer to the output
    finalSoftmax.inputConnect(1,1) = 1;
    finalSoftmax.outputConnect = 1;

    % Add a connection for a bias term to the first layer
    finalSoftmax.biasConnect = 1;

    % Set the size of the input and the 1st layer
    finalSoftmax.inputs{1}.size = inputSize;
    finalSoftmax.layers{1}.size = outputSize;

    % Use the softmax transfer function for the first layer
    finalSoftmax.layers{1}.transferFcn = 'softmax';

    % Use all of the data for training
    finalSoftmax.divideFcn = 'dividetrain';

    % Use the cross-entropy performance function
    finalSoftmax.performFcn = 'crossentropy';

    % number of training epochs and the training function
    finalSoftmax.trainFcn = 'trainscg'; % scaled conjugate gradient method
    finalSoftmax.trainParam.epochs = epochs;

end % function