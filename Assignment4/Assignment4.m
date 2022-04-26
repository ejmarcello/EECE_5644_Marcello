%% Assignment 4
% Assignment4.m
%
% REQUIRES: Image Processing Toolbox
%
% by Ethan Marcello
%           initialization date: 21 APR 2022
%% Question 1 (Data Distribution)
% reusing code I wrote for Assignment 1 question 2.
clear all; close all;

% Setup: create N(i) samples based on data distribution & keep track of
% class labels.
N = [1000 10000]; % numbers of samples for train then test dataset(s)
n = 2; % dimensionality of the data vector
C = 2; % Number of classes

priors = [0.5 0.5]; % assumed class data occurs with equal probability
r = [2 4]; % coefficient determining class r(c).

for i = 1:length(N)
    data(i).x = [];
    data(i).labels = [];
    for c = 1:C
        Ns = N(i)*priors(c); % number of samples to generate
        xnew = r(c).*[cos((2.*pi.*rand(1,Ns))-pi);sin((2.*pi.*rand(1,Ns))-pi)] +...
                randGaussian(Ns,[0;0],[1 0;0 1]); %randn(2,Ns);
        data(i).x = [data(i).x xnew];
        data(i).labels = [data(i).labels c.*ones(1,Ns)];
    end     
end

% class labels are "1" and "2" for "-1" and "+1" respectively.

% Plot data in 2D to see level of overlap
figure;
Nc1 = N(1)*priors(1); 
plot(data(1).x(1,1:Nc1),data(1).x(2,1:Nc1),'r*')
hold on
plot(data(1).x(1,Nc1+1:end),data(1).x(2,Nc1+1:end),'g*')
title("Data Samples")

clear c i Nc1 Ns xnew;
% Save this dataset to use for later.
%save('A4Q1Dataset');

%% Train MLP for Question 1
clear all; close all;
% load in data 
load("A4Q1Dataset.mat");
N_vector = N;
numTest = 10; % # of final MLP iterations w/ test data to ensure maximum performance.
    

x = data(1).x; 
labels = data(1).labels;
N = N_vector(1); % number of data samples

% First get the test data (this is used to check for accuracy later)
XTest = data(2).x'; % these are both transposed to work with deep learning tools
YTest = data(2).labels';
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
p = round(linspace(10,18,9));
n = size(XTest,2); 
% size of output... equal to number of classes
C = 2;
% m = 6;
% i = 1;
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
    % quadratic activation function followed by softmax at the
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
    initialLearnRate = 0.85;
    maxEpochs = 12;
    miniBatchSize = 64;
    validationFrequency = 20;
    % uses stochastic gradient with momentum solver... but set momentum to 0.
    options = trainingOptions("sgdm", ...
        InitialLearnRate=initialLearnRate, ...
        MaxEpochs=maxEpochs, ...
        MiniBatchSize=miniBatchSize, ...
        Momentum=0.2, ...
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
    error = sum(error_mi,2)./K; % yields a Mx1 vector sample-able by 'm' where M is the number of models tested
    %%%%%%%%%%% MAKE PLOT %%%%%%%%%%%%
    YPredtest = classify(net,XTest,'MiniBatchSize',miniBatchSize);
    YPredtest = double(YPredtest); % convert to double
    idxc = []; idxw = []; % indices of correct and wrong classifications
    for id = 1:length(YPredtest)
       if YPredtest(id) == data(2).labels(id)
           idxc = [idxc id];
       else
           idxw = [idxw id];
       end
    end
    c1 = find(idxc<=5000); w1 = find(idxw<=5000); % correct and wrong decisions for class 1
    c2 = find(idxc>5000); w2 = find(idxw>5000); % correct and wrong decisions for class 2

    figure(m);
    plot(data(2).x(1,idxc(c2)),data(2).x(2,idxc(c2)),'g*')
    hold on
    plot(data(2).x(1,idxc(c1)),data(2).x(2,idxc(c1)),'c+')
    plot(data(2).x(1,idxw(w2)),data(2).x(2,idxw(w2)),'r*')
    plot(data(2).x(1,idxw(w1)),data(2).x(2,idxw(w1)),'y+')
    title(['MLP Classifications With ' num2str(p(m)) ' Perceptrons'])
    legend('Correct Class l=+1','Correct Class l=-1','Incorrect Class l=+1','Incorrect Class l=-1')

end



%% Test 2

%---Test script section to see generated classification boundary---%
% Or more easily I can show classification decisions...
figure;
Nc1 = N_vector(2)*priors(1); 
plot(data(2).x(1,Nc1+1:end),data(2).x(2,Nc1+1:end),'g*')
hold on
plot(data(2).x(1,1:Nc1),data(2).x(2,1:Nc1),'c+')
%legend('Class r_+','Class r_-')
% create grid over entire space
x1range = min(XTest(:,1)):.1:max(XTest(:,1));
x2range = min(XTest(:,2)):.1:max(XTest(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

predictedvalues = classify(net,XGrid,'MiniBatchSize',miniBatchSize);
predictedvalues = double(predictedvalues);
%figure;
   gscatter(xx1(:), xx2(:), predictedvalues,'rn');

   title('Classification Regions')
   
   axis([-8 10 -9 9])
legend('Class l=+1','Class r=-1','Classification Region for l=-1','Location','southeast')

%% Re-train chosen MLP model.
% Now choose 'least rejectable' model
[~,mstar] = min(error); pstar = p(mstar);
% Train this model 10 times with random initializations and take the best result.
for i = 1:numTest % test it numTest times and take best performing network.
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
    Verbose=false, ...
    Plots='none'); % no validation included, don't show training-progress

model(i).net = trainNetwork(XTrain,YTrain,layers,options); % final network
%XTest and YTest already defined at top of section.
YPred = classify(model(i).net,XTest,'MiniBatchSize',miniBatchSize);
% 
model_errors(i) = 1 - (sum(YPred == categorical(YTest))/numel(YTest)); % final error
end

[besterr,bestidx] = min(model_errors);
bestmodel.net = model(bestidx).net; % saving the best network structure based on data in dsn
numPerceptrons = pstar;



%% Run Theoretical classifier for Question 1
%clear all; close all
% x = dataset(7).x; 
% labels = dataset(7).labels;
% N = N_vector(7);
% 
% %%% Classification with True Data PDF (benchmark)
% %%%%% MAP Classification rule with true data PDF %%%%%
% % gmmnum is the number of the gaussian in the mixture model.
% for gmmnum = 1:length(gmmParameters.meanVectors(1,:))
%     pxgivenl(gmmnum,:) = evalGaussianPDF(x,gmmParameters.meanVectors(:,gmmnum),gmmParameters.covMatrices(:,:,gmmnum)); % Evaluate p(x|L=GMM_number)
% end
% 
% px = gmmParameters.priors*pxgivenl; % Total probability theorem
% classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1); % P(C=l|x)
% 
% % Finally make classification decisions
% lossMatrix = ones(C)-eye(C); % For min-Perror design, use 0-1 loss
% expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
% [~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP
% 
% % Find empirically estimated probability of error
% pError = sum(length(find(labels~=decisions)))/N;
% clear x labels;
%% Creating the MLP to classify data 
% Continues straight from "(benchmark)" section

%% Plotting final results for Question 1

figure;
semilogx(N_vector(1:6),pError*ones(1,6),'r-','LineWidth',1.5);
hold on
semilogx(N_vector(1:6),besterr,'k-','LineWidth',1.5);
xlabel('Number of Samples'); ylabel('Error');
title('Empirically Estimated Classification Errors');
legend('Theoretical Optimum','MLP Trained Optimum');
grid on
axis([100 10000 0.1 0.35]);
ax = gca;
ax.FontSize = 18;
% Model trained with cross validation and then network was 

figure;
semilogx(N_vector(1:6),numPerceptrons,'*','MarkerSize',12,'LineWidth',1.5);
xlabel('Number of Samples'); ylabel('Number of Perceptrons');
title('Optimal Number of Perceptrons Used vs. Sample Size');
grid on
ax = gca;
ax.FontSize = 18;
ax.YTick = 5:15;


%% BEGIN QUESTION 2 HERE
%clear all; close all;

% Berkeley Segmentation Dataset: Training Image #247085 [color]
% Image of a rough collie in tall grass with blue cloudy sky bkgnd and fence post
I = imread('../Datasets/Collie.jpg');
I = im2double(I); % converts to double precision and normalizes to 1
I = imresize(I,0.25);
%feature vectors: row index, col index, R val, G val, B val (5 features per
%pixel)
% MAY want to downsample image first... Seems that the GMM model fitting
% takes quite a long time to do 10 fold Cross Validation on models with order
% greater than 1

rowsize = size(I,1);
colsize = size(I,2);
x = zeros(5,size(I,1)*colsize); % initialize empty
for row = 1:rowsize
    for col = 1:colsize
        samp = ((row-1)*colsize)+col;
        x(:,samp) = [row col I(row,col,1) I(row,col,2) I(row,col,3)]';
    end
end
x(1,:) = x(1,:)./max(x(1,:)); % divide by the max to normalize row feature to 1
x(2,:) = x(2,:)./max(x(2,:)); % " for column feature.

clear col colsize row rowsize samp;

n = size(x,1);
N = size(x,2);
% Save this dataset to use for later.
%save('A4Q2Dataset');

%% GMM fitting with 10 fold Cross Validation
%clear all; close all;

%load('A4Q2Dataset.mat');
%Fit a Gaussian Mixture Model to these normalized feature vectors representing
%the pixels of the image. To fit the GMM, use maximum likelihood parameter 
%estimation and 10-fold crossvalidation (with maximum average 
%validation-log-likelihood as the objective) for model order selection.

% Goal for GMM fit is to maximize the log likelihood of 1 gaussians, 
% 2 gaussians, 3 gaussians, etc. against the data. The one that maximizes
% the most "wins" as the best GMM fit.

% This code was reused from Assignment 3, but edited for the new data.
X = x; % save untouched data in X
M = 10; % total number of gaussians to test the fit
K = 10; % for 10-fold cross validation
nsamp = ceil(N/K); % number of samples to take for each iteration
numOfExperiments = 1; % can re-run the model order selection this number of times.

% preallocate space
logL = zeros(1,M);
logLikelihood = zeros(1,K);
modelSel = zeros(1,numOfExperiments); 

for exp = 1:numOfExperiments
%warns(exp).msg = [];
    % Taken from Matlab documentation: https://www.mathworks.com/help/stats/fit-a-gaussian-mixture-model-to-data.html?searchHighlight=Gaussian%20mixture%20model%20fit&s_tid=srchtitle_Gaussian%20mixture%20model%20fit_1
 
    x = x'; % fitgmdist likes data in columns, so this is transposed
    idx = randperm(size(x,1),N); % randomly mix up the data 
    x = x(idx,:);

    % GMM fitting options
    options = statset('MaxIter',1000,'Display','final'); % can also specify 'Display','final' to show num iterations and log-likelihood
    % 'Start','randSample' will randomly choose K starting points for the
    % gaussians.
    for m = 4:M %for each model fit (number of gaussians to fit to data)
        for i = 1:K
            %%%--- Get Train and Validation sets ready for iteration i ---%%%
            idxvs = ((i-1)*nsamp+1); idxve = idxvs+nsamp-1;
            if i == K % if on last iteration
                idxve = N; % set the end to the last data sample.
            end
            idxv = idxvs:idxve; % validation indices (idx) from start to end
            XTrain = x;
            XValidation = XTrain(idxv,:); % now that data is transposed, sample like this
            XTrain(idxv,:) = []; % removes validation samples from train data.

            % set a warning trap (reset step)
            %lastwarn('','');
            % Fit the GMM using EM algorithm.
            gm{m} = fitgmdist(XTrain,m,'RegularizationValue',0.01,'Options',options); % The cell {m} is overwritten in each i:K loop
            %[warnMsg,warnId] = lastwarn(); % grab to see if there is a warning here
            % tracks error messages over the experiments.
%             if (~isempty(warnId)) % If there was a warning, store it.
%                 disp('There was a warning message fitting the Gaussian mixture model');
%                 warns(exp).msg = [warns(exp).msg; ...
%                     strcat("For sample size N=", num2str(N), " model fit m=", num2str(m), ", iteration ",...
%                     num2str(i), ": ", warnMsg)];
%             end
            % Validate by calculating the log likelihood
            % calculate the log likelihood - sampled from class EMforGMM.m script
            alpha = gm{m}.ComponentProportion'; % mixing coefficients
            logLikelihood(i) = sum(log(evalGMM(XValidation',alpha,gm{m}.mu',gm{m}.Sigma)));
        end % end K-fold C.V.
        % calculate average validation "error" for model m which is 
        % log likelihood in this case. Maximum one of these will be the "best" fit.
        logL(m) = sum(logLikelihood)./K;
    end % end models

    [~,bestm] = max(logL);
    modelSel(exp) = bestm; % experiment numbers stored in columns, datasets on the rows


end % end of experiments

save('A4Q2modelFitting.mat'); % only thing you need from this is the "bestm"
                % i.e. the best model order. Determines number of segments
                % to use in the next section.

%% Plots, data processing, re-fitting, contour and scatter plots 

%load('A4Q2modelFitting.mat');
bestm = 20; % overriding work to choose number of segments you want.
 
%%% re-fit GMM several times with best model
for iter = 1:10
    gm{iter} = fitgmdist(x,bestm,'RegularizationValue',0.01,'Options',options);
    negLogL(iter) = gm{iter}.NegativeLogLikelihood;
end
[~,bestiter] = max(-1.*negLogL);
bestgm = gm{bestiter}; 

%%
%%%%-----Now classify samples using bestgm-----%%%%
% use the untouched "X"; i.e., the properly ordered one.
if size(X,1) > size(X,2)
    X = X';
end

gmmParameters.priors = bestgm.ComponentProportion;
gmmParameters.meanVectors = bestgm.mu';
gmmParameters.covMatrices = bestgm.Sigma;

%%%%% MAP Classification rule with fitted data PDF %%%%%
% gmmnum is the number of the gaussian in the mixture model.
for gmmnum = 1:length(gmmParameters.meanVectors(1,:))
    pxgivenl(gmmnum,:) = evalGaussianPDF(X,gmmParameters.meanVectors(:,gmmnum),gmmParameters.covMatrices(:,:,gmmnum)); % Evaluate p(x|L=GMM_number)
end

px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,bestm,1); % P(L=l|x)

% Finally make classification decisions
lossMatrix = ones(bestm)-eye(bestm); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

rowsize = size(I,1);
colsize = size(I,2);
gray = linspace(0,1,bestm); % grayscale levels
J = zeros(rowsize,colsize); %preallocate space for segmented image
for row = 1:rowsize
    for col = 1:colsize
        samp = ((row-1)*colsize)+col;
        J(row,col) = gray(decisions(samp)); %assign grayscale level to decisions in the image.
    end
end
montage({I,J}); % displays the original (downsampled) image and the segmented one side by side.
clear pxgivenl px classPosteriors decisions gmmParameters ...
        expectedRisks lossMatrix
%% Utility Functions

%%% Taken from class script "EMforGMM.m" ...
function gmm = evalGMM(x,alpha,mu,Sigma)
% EVALGMM.M - Evaluates the gmm surface (sums values of each gaussian pdf)
% takes x as input nxN matrix where num of samples is N
% mu is matrix of mean vectors as column vectors.
% Sigma is 3D array of stacked nxn covariance matrices.
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    % the following yields a 1xN vector containing the summed value of each
    % sample in each pdf. Should -not- be normalized.
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m)); % calls evalGaussian function (below)
end
end

%%% Taken from class script "EMforGMM.m" ...
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
% mu is a single column vector.
% Sigma is an nxn covariance matrix
[n,N] = size(x); % again x is an nxN input. n is dimension, N is num of samples
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1); % sums all the rows together leaving a 1xN vector
g = C*exp(E); % g is a scaled 1xN vector (C is the coefficient for the complex exponential of the multivariate Gaussian pdf equation)
end
