%% function [imagesMatrix,inputSize] = imagesVecMatrix(Images)
% desc: Turn the training images into vectors and put them in a matrix
% inputs: vector of images(i.e. 28x28)
% output: 
% =====================================================
function [imagesMatrix,inputSize] = imagesVecMatrix(Images)

% size of each image(vectorized)
inputSize = numel(Images{1}(:)); 

% Turn the training images into vectors and put them in a matrix
imagesMatrix = zeros(inputSize, numel(Images));
for i = 1:numel(Images)
    imagesMatrix(:,i) = Images{i}(:);
end

end % function