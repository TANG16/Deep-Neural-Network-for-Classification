%% function viewImages(Images,rows,cols)
% desc: Display some images from Image matrix in a plot
% inputs: 
% output: 
% =====================================================
function viewImages(Images,rows,cols)

% Display some images of image matrix Images
clf % clear current figure window
n = rows*cols;
for i = 1:n
    subplot(rows,cols,i);
    imshow(Images{i});
end

end % function