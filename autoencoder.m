%% function autoenc = autoencoder(xInput, hiddenSize,epochs)
% desc: autoencoder network
% inputs: 
% output: 
% =====================================================
function autoenc = autoencoder(xInput,hiddenSize,epochs)

    % Create the network. 
    autoenc = feedforwardnet(hiddenSize); % type of network
    autoenc.trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation
    autoenc.trainParam.epochs = epochs;

    % Do not use process functions at the input or output
    autoenc.inputs{1}.processFcns = {};
    autoenc.outputs{2}.processFcns = {};

    % Set the transfer function for both layers to the logistic sigmoid
    autoenc.layers{1}.transferFcn = 'logsig';
    autoenc.layers{2}.transferFcn = 'logsig';

    % Use all of the data for training
    autoenc.divideFcn = 'dividetrain';

    %sparse representation
    autoenc.performFcn = 'msesparse';
    
    % experiment by altering these parameters
	autoenc.performParam.L2WeightRegularization = 0.002; %0.002-0.004;
    autoenc.performParam.sparsityRegularization = 4; %4; 
    autoenc.performParam.sparsity = 0.1; %0.1-0.15;

    % Train the autoencoder
	xOutput = xInput;
    autoenc = train(autoenc,xInput,xOutput); % same data for input and output

end % function
