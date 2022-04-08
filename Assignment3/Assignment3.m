%% Assignment 3
% Assignment3.m

%
% by Ethan Marcello
%
%% Question 1 (Data Distribution)
% reusing code I wrote for Assignment 1 question 2.
clear all; close all;

% Setup: create 10k samples based on data distribution & keep track of
% class labels.
N = [100 200 500 1000 2000 5000 100000]; % numbers of samples for ith dataset
n = 3; % dimensionality of the data vector
sep = 1.25; % Data mean separation parameter
C = 4; % Number of classes
gmmParameters.priors = [0.25 0.25 0.25 0.25]; % equal priors
% Construct data means by placing them on the corners of a cube
m1 = zeros(n,1); % mean vector for C=1. Length is same as n
m2 = [sep 0 0]'; % mean vector for C=2
m3 = [0 sep 0]'; % mean vector for C=3
m4 = [0 0 sep]'; % mean vector for C=4
gmmParameters.meanVectors = [m1 m2 m3 m4];

% Now calculate Covariance matricies.
A1 = rand(n,n);
A2 = rand(n,n);
A3 = rand(n,n);
A4 = rand(n,n);
s1 = rand()*0.3;
s2 = rand()*0.3;
s3 = rand()*0.3;
s4 = rand()*0.3;
%Sigma = (I+sA)(I+sA)' with 0<s<<1
Cov1 = 0.2.*(eye(3)+s1.*A1)*(eye(3)+s1.*A1)';
Cov2 = 0.2.*(eye(3)+s2.*A2)*(eye(3)+s2.*A2)';
Cov3 = 0.2.*(eye(3)+s3.*A3)*(eye(3)+s3.*A3)';
Cov4 = 0.2.*(eye(3)+s4.*A4)*(eye(3)+s4.*A4)';
gmmParameters.covMatrices(:,:,1) = Cov1;
gmmParameters.covMatrices(:,:,2) = Cov2;
gmmParameters.covMatrices(:,:,3) = Cov3;
gmmParameters.covMatrices(:,:,4) = Cov4;
clear A1 A2 A3 A4 s1 s2 s3 s4;

for i = 1:length(N)
      % Using class priors to generate data x
    x1 = randGaussian(N(i)*0.25,m1,Cov1); % C=1 data, Uses randGaussian function
    x1(:,:,2) = ones(n,N(i)*0.25); % Add label to 3rd dimension of mtx
    x2 = randGaussian(N(i)*0.25,m2,Cov2); % C=2 data, Uses randGaussian function
    x2(:,:,2) = 2.*ones(n,N(i)*0.25); % Add label to 3rd dimension
    x3 = randGaussian(N(i)*0.25,m3,Cov3); % C=3 data, first gaussian model
    x3(:,:,2) = 3.*ones(n,N(i)*0.25); % Add label to 3rd dimension
    x4 = randGaussian(N(i)*0.25,m4,Cov4); % C=3 data, second gaussian model
    x4(:,:,2) = 4.*ones(n,N(i)*0.25); % Add label to 3rd dimension

    x = [x1(:,:,:) x2(:,:,:) x3(:,:,:) x4(:,:,:)]; 
    % We know all C=1 data is in x1 and all C=2 data is in x2, etc.
    labels = x(1,:,2);
    x = x(:,:,1);
    dataset(i).x = x;
    dataset(i).labels = labels;
end

clear m1 m2 m3 m4 Cov1 Cov2 Cov3 Cov4;

% Plot data in 3D to see level of overlap
figure;
plot3(x1(1,:)',x1(2,:)',x1(3,:)','r*')
hold on
plot3(x2(1,:)',x2(2,:)',x2(3,:)','g*')
plot3(x3(1,:)',x3(2,:)',x3(3,:)','c*')
plot3(x4(1,:)',x4(2,:)',x4(3,:)','m*')
title("Data Samples")
clear x1 x2 x3 x4;

% Save this dataset to use for later.
% save('A3Q1Dataset');

%% Classification with True Data PDF (benchmark)
clear all; close all;

% load in data 
load('A3Q1Dataset.mat');
dsn = 3; % choose dataset number
x = dataset(dsn).x; % choose dataset D (xxxx samples)
labels = dataset(dsn).labels;
N = N(dsn); % number of data samples

%%%%% MAP Classification rule with true data PDF %%%%%
% gmmnum is the number of the gaussian in the mixture model.
for gmmnum = 1:length(gmmParameters.meanVectors(1,:))
    pxgivenl(gmmnum,:) = evalGaussianPDF(x(:,:,1),gmmParameters.meanVectors(:,gmmnum),gmmParameters.covMatrices(:,:,gmmnum)); % Evaluate p(x|L=GMM_number)
end

px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1); % P(C=l|x)

% Finally make classification decisions
lossMatrix = ones(C)-eye(C); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% Find empirically estimated probability of error
pError = sum(length(find(labels~=decisions)))/N;

%% Creating the MLP to classify data (estimates class posterior probabilities)
% Continues straight from "(benchmark)" section

% First get the test data (this is used to check for accuracy later)
XTest = dataset(7).x'; % these are both transposed to work with deep learning tools
YTest = dataset(7).labels';
% NOW Setup the data for cross-validation by partitioning.
XTrain = x'; % network input features
YTrain = categorical(labels'); % network output responses (classes)
idx = randperm(size(XTrain,1),N); % mix up the data 
XTrain = XTrain(idx,:);
YTrain = YTrain(idx); % now data is randomly mixed around. Ready to partition.

% K-fold cross validation. Determines number of attempts/data partitions.
K = 10;
% Structure of MLP: Only one hidden layer, one output layer
% p = round(logspace(0,3,10)); 
p = round(linspace(2,15,14));
n = size(XTest,2); 
% size of output... equal to number of classes
C = 4;
for m = 1:length(p)
    for i = 1:K
        % if bootstrapping is desired, then ...
        % can sample with replacement using randi(N,1,K). This does sample
        % without replacement using randperm(N,K)
    %%-- Select Validation and Test Data for run i --%%
    idxvs = ((i-1)*(N/K)+1); idxve = idxvs+(N/K)-1;
    idxv = idxvs:idxve; % validation indices (idx) from start to end
    XTraini = XTrain;
    YTraini = YTrain;
    XValidation = XTrain(idxv,:); % now that data is transposed, sample like this
    XTraini(idxv,:) = []; % removes validation samples from train data.
    YValidation = YTrain(idxv);
    YTraini(idxv) = [];

    % Randomly initialize parameter estimates for C outputs. Results of MLP will be
    % posterior probabilities (using softmax @ the output layer)
    randscale = 0.5; % arbitrarily chosen scaling factor to bring initialized parameters closer to zero.
    theta.b2 = randscale*rand(C,1); theta.W2 = randscale*rand(C,p(m)); % output layer params
    theta.b1 = randscale*rand(p(m),1); theta.W1 = randscale*rand(p(m),n); % 1st hidden layer params

    %%%%%--- Deep Learning Toolbox, create custom MLP ---%%%%%
    % Can use Deep Learning Toolbox to build a custom MLP.
    % uses function trainNetwork() to train data. Since this is a
    % classification problem, I will use data as Nxn matrix input with
    % 'responses' being an Nx1 vector of true class labels. The network will be
    % defined by the layer array 'layers' containing fullyConnected layers and
    % softplus (smooth ReLU) activation function followed by softmax at the
    % output. Finally a classification is given at the output using a
    % classification layer that minimizes cross entropy loss.


    %%% now to build the network layers %%%
    layer_fc1 = fullyConnectedLayer(p(m),'Name','fc1','Weights',theta.W1,...
                    'Bias',theta.b1); % with p neurons and
    % weights must be specified by px{inputsize} matrix.
    layer_fc2 = fullyConnectedLayer(C,'Name','fc2','Weights',theta.W2,...
                    'Bias',theta.b2); % with C neurons for class outputs
    layers = [...
              featureInputLayer(n)
              layer_fc1
              softplusLayer
              layer_fc2
              softmaxLayer 
              classificationLayer]; % Using the classification Layer "hides" 
      % the posterior probabilities, but classifies the samples using minimum
      % cross entropy loss which is what I need.

    %%% specify the training options %%%
    initialLearnRate = 0.5;
    maxEpochs = 12;
    miniBatchSize = 64;
    validationFrequency = 20;
    % uses stochastic gradient with momentum solver... but set momentum to 0.
    options = trainingOptions("sgdm", ...
        InitialLearnRate=initialLearnRate, ...
        MaxEpochs=maxEpochs, ...
        MiniBatchSize=miniBatchSize, ...
        Momentum=0, ...
        ValidationData={XValidation,YValidation}, ...
        ValidationFrequency=validationFrequency, ...
        Verbose=false, ...
        Plots='none');
%         Plots="training-progress");

    % Train the network
    net = trainNetwork(XTraini,YTraini,layers,options);
    % calculate error metric
    YPred = classify(net,XValidation,'MiniBatchSize',miniBatchSize);
    error_mi(m,i) = 1 - (sum(YPred == YValidation)/numel(YValidation));
    end
    % Calculate the average error for each model m here.
    error = sum(error_mi,2)./K; % yields a vector sample-able by 'm'
end

% Now choose 'least rejectable' model
[~,mstar] = min(error);
% Retrain model using mstar on all training data
% First randomly initialize weights with mstar
theta.b2 = randscale*rand(C,1); theta.W2 = randscale*rand(C,p(mstar)); % output layer params
theta.b1 = randscale*rand(p(mstar),1); theta.W1 = randscale*rand(p(mstar),n); % 1st hidden layer params
%%% now to build the network layers %%%
layer_fc1 = fullyConnectedLayer(p(mstar),'Name','fc1','Weights',theta.W1,...
                'Bias',theta.b1); % with p neurons and
% weights must be specified by px{inputsize} matrix.
layer_fc2 = fullyConnectedLayer(C,'Name','fc2','Weights',theta.W2,...
                'Bias',theta.b2); % with C neurons for class outputs
layers = [...
          featureInputLayer(n)
          layer_fc1
          softplusLayer
          layer_fc2
          softmaxLayer 
          classificationLayer];  
% uses stochastic gradient with momentum solver... but set momentum to 0.
options = trainingOptions("sgdm", ...
    InitialLearnRate=initialLearnRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Momentum=0, ...
    Verbose=true, ...
    Plots='training-progress'); % no validation included, show training progress

net_fin = trainNetwork(XTrain,YTrain,layers,options); % final network
%XTest and YTest already defined at top of section.
YPred = classify(net_fin,XTest,'MiniBatchSize',miniBatchSize);
% 
error_fin = 1 - (sum(YPred == categorical(YTest))/numel(YTest)) % final error

