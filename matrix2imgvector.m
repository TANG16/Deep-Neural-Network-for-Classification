%% function imgvector = matrix2imgvector(ImageMatrix,imageHeight,imageWidth)
% desc: convert an image matrix(i.e. 784x5000) to an image vector of
% (1x5000)
% inputs: 
% output: 
% =====================================================
function imgvector = matrix2imgvector(ImageMatrix,imageHeight,imageWidth)

[m,n] = size(ImageMatrix);

for i = 1:n
  imgvector{i} = reshape(ImageMatrix(:,i),[imageHeight,imageWidth]);
end

end % function