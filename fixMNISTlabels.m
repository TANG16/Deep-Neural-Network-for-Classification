%% function viewImages(Images,rows,cols)
% desc: Display some images from Image matrix in a plot
% inputs: 
% output: 
% =====================================================
function fixedLabels = fixMNISTlabels(Labels)


[m,n] = size(Labels);

for i = 1:m
    for j = 1:9
        if (Labels(i) == 0)  % label 0 in slot 10
            fixedLabels(10,i) = 1;
        elseif (Labels(i) == j)
            fixedLabels(j,i) = 1;
        else
            fixedLabels(j,i) = 0;
        end
    end % j
end % i

end % function