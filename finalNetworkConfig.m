%% function finalNetwork = finalNetworkConfig()
% desc: final network configuration
% inputs: 
% output: 
% 
% =====================================================
% 
function finalNetwork = finalNetworkConfig(autoencHid1,...
                                           autoencHid2,...
                                           finalSoftmax,...
                                           inputSize,...
                                           hiddenSize1,...
                                           hiddenSize2,...
                                           outputSize,...
                                           epochs)

    
    % Create an empty network
    finalNetwork = network;

    % Specify one input and three layers
    finalNetwork.numInputs = 1;
    finalNetwork.numLayers = 3;

    % Connect the 1st layer to the input
    finalNetwork.inputConnect(1,1) = 1;

    % Connect the 2nd layer to the 1st layer
    finalNetwork.layerConnect(2,1) = 1;

    % Connect the 3rd layer to the 2nd layer
    finalNetwork.layerConnect(3,2) = 1;

    % Connect the output to the 3rd layer
    finalNetwork.outputConnect(1,3) = 1;

    % Add a connection for a bias term for each layer
    finalNetwork.biasConnect = [1; 1; 1];

    % Set the size of the input
    finalNetwork.inputs{1}.size = inputSize;

    % Set the size of the first layer to the same as the layer in autoencHid1
    finalNetwork.layers{1}.size = hiddenSize1;

    % Set the size of the second layer to the same as the layer in autoencHid2
    finalNetwork.layers{2}.size = hiddenSize2;

    % Set the size of the third layer to the same as the layer in finalSoftmax
    finalNetwork.layers{3}.size = outputSize;

    % Set the transfer function for the first layer to the same as in
    % autoencHid1
    finalNetwork.layers{1}.transferFcn = 'logsig';

    % Set the transfer function for the second layer to the same as in
    % autoencHid2
    finalNetwork.layers{2}.transferFcn = 'logsig';

    % Set the transfer function for the third layer to the same as in
    % finalSoftmax
    finalNetwork.layers{3}.transferFcn = 'softmax';

    % Use all of the data for training
    finalNetwork.divideFcn = 'dividetrain';

    % Copy the weights and biases from the three networks that have already
    % been trained
    finalNetwork.IW{1,1} = autoencHid1.IW{1,1};
    finalNetwork.b{1} = autoencHid1.b{1,1};
    finalNetwork.LW{2,1} = autoencHid2.IW{1,1};
    finalNetwork.b{2} = autoencHid2.b{1,1};
    finalNetwork.LW{3,2} = finalSoftmax.IW{1,1};
    finalNetwork.b{3} = finalSoftmax.b{1,1};

    % Use the cross-entropy performance function
    finalNetwork.performFcn = 'crossentropy';

    % You can experiment by changing the number of training epochs and the
    % training function
    finalNetwork.trainFcn = 'trainscg';
    finalNetwork.trainParam.epochs = epochs;

end % function