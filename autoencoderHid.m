%% function autoencHid = autoencoderHid(autoenc,inputSize,outputSize)
% desc: create autoencoder hidden layer
% inputs: 
% output: 
% =====================================================
function autoencHid = autoencoderHid(autoenc,inputSize,outputSize)

    % Create an empty network
    autoencHid = network;

    % Set the number of inputs and layers
    autoencHid.numInputs = 1;
    autoencHid.numlayers = 1;

    % Connect the 1st (and only) layer to the 1st input, and also connect the
    % 1st layer to the output
    autoencHid.inputConnect(1,1) = 1;
    autoencHid.outputConnect = 1;

    % Add a connection for a bias term to the first layer
    autoencHid.biasConnect = 1;

    % Set the size of the input and the 1st layer
    autoencHid.inputs{1}.size = inputSize;
    autoencHid.layers{1}.size = outputSize;

    % Use the logistic sigmoid transfer function for the first layer
    autoencHid.layers{1}.transferFcn = 'logsig';

    % Copy the weights and biases from the first layer of the trained
    % autoencoder to this network
    autoencHid.IW{1,1} = autoenc.IW{1,1};
    autoencHid.b{1,1} = autoenc.b{1,1};

end % function