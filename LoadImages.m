


% % load train and test MNIST Background Images data
% xTrain = load('./data/mnist_background_images/mnist_background_images_train.amat');
% xTest = load('./data/mnist_background_images/mnist_background_images_test.amat');
% tTrain = xTrain(:, 785);
% tTest = xTest(:, 785);
% xTrain = xTrain(:,1:784)';
% xTest = xTest(:,1:784)';
% save('./data/mnist_background_images/mnist_background_images');
% load('./data/mnist_background_images/mnist_background_images');


% % load train and test MNIST Background random data
% xTrain = load('./data/mnist_background_random/mnist_background_random_train.amat');
% xTest = load('./data/mnist_background_random/mnist_background_random_test.amat');
% tTrain = xTrain(:, 785);
% tTest = xTest(:, 785);
% xTrain = xTrain(:,1:784)';
% xTest = xTest(:,1:784)';
% save('./data/mnist_background_random/mnist_background_random');
% load('./data/mnist_background_random/mnist_background_random');

% % load train and test MNIST rotation data
% xTrain = load('./data/mnist_rotation/mnist_rotation_train.amat');
% xTest = load('./data/mnist_rotation/mnist_rotation_test.amat');
% tTrain = xTrain(:, 785);
% tTest = xTest(:, 785);
% xTrain = xTrain(:,1:784)';
% xTest = xTest(:,1:784)';
% save('./data/mnist_rotation/mnist_rotation');
% load('./data/mnist_rotation/mnist_rotation');

% % load train and test MNIST background and rotation data
% xTrain = load('./data/mnist_background_rotation/mnist_background_rotation_train.amat');
% xTest = load('./data/mnist_background_rotation/mnist_background_rotation_test.amat');
% tTrain = xTrain(:, 785);
% tTest = xTest(:, 785);
% xTrain = xTrain(:,1:784)';
% xTest = xTest(:,1:784)';
% save('./data/mnist_background_rotation/mnist_background_rotation');
% load('./data/mnist_background_rotation/mnist_background_rotation');

% load train and test MNIST basic
xTrain = load('./data/mnist_basic/mnist_train.amat');
xTest = load('./data/mnist_basic/mnist_test.amat');
tTrain = xTrain(:, 785);
tTest = xTest(:, 785);
xTrain = xTrain(:,1:784)';
xTest = xTest(:,1:784)';
save('./data/mnist_basic/mnist_basic');
load('./data/mnist_basic/mnist_basic');
