% desc: compressed version of vector representation
% inputs: vector of images
% output: 
% =====================================================
function [imgComp] = compressedVecMatrix(imgMatrix)

    [m,n] = size(imgMatrix);
        
    max_vector = max(sum(imgMatrix)); % maximum vector length
        
    mDist = (ones(n,1)*(1:m))'; % matrix of distances
        
    mComp = imgMatrix .* mDist; % add indexes to matrix
        
    imgComp = zeros(max_vector,n); % create final matrix
        
    for j = 1:n % columns
        idx = 1;
        for i = 1:m % rows
            if (mComp(i,j) > 0)
                imgComp(idx,j) = mComp(i,j);
                idx = idx+1;
            end
        end %j
    end %i
    
   
end % function