%% Assignment2.m Script for the second ML&PR Assignment
%
% By Ethan Marcello 21 March 2022
%
%% Question 1 setp (create datasets)

% Setup: create 10k samples based on data distribution & keep track of
% class labels.
N = [20 200 2000 10000]; % 4 diff datasets of sample numbers N(i)
n = 2; % dimensionality of the data vector
m01 = [3 0]'; % mean vector for L=0, GMM 1 
m02 = [0 3]'; 
m1 = [2 2]'; % mean vector for L=1
gmmParameters.meanVectors = [m01 m02 m1];
w1 = 0.5; % GMM weights for class L=0 distribution
w2 = 0.5; 

C01 = [2 0;
       0 1];
C02 = [1 0;
       0 2];
C1 = eye(2);
gmmParameters.covMatrices(:,:,1) = C01;
gmmParameters.covMatrices(:,:,2) = C02;
gmmParameters.covMatrices(:,:,3) = C1;

% class priors
pL0 = 0.65;
pL1 = 0.35;
gmmParameters.priors = [pL0 pL1];

for i = 1:length(N)
      picks0 = rand([1,N(i)*pL0]); % generate a random way to choose the samples for L=0
      num01 = sum(picks0>0.5); % number of samples for x01
      num02 = (N(i)*pL0)-num01;
      % Using class priors to generate data x
      x01 = randGaussian(num01,m01,C01); % L=0 data GMM 1, Uses randGaussian function
      x02 = randGaussian(num02,m02,C02); % L=0 data GMM 2, Uses randGaussian function
      x1 = randGaussian(N(i)*pL1,m1,C1); % L=1 , Uses randGaussian function
      
      data(i).x = [x01 x02 x1]; % All L=0 data is in x0 and all L=1 data is in x1
      
      x0_label = zeros(1,N(i)*pL0); % create labels for first section      
      x1_label = ones(1,N(i)*pL1); % create labels for second section
      data(i).labels = [x0_label x1_label];
      
end

% Save this dataset to use for all cases.
%save('A2Q1Dataset');

%% Question 1 (Part 1)
%  ERM classification using the knowledge of true data pdf
clear all; close all;
load('A2Q1Dataset');

% gmmnum is the number of the gaussian in the mixture model.
for gmmnum = 1:length(gmmParameters.meanVectors(1,:))
    % data(4) has the 10k samples
    pxgivenGMM(gmmnum,:) = evalGaussianPDF(data(4).x,gmmParameters.meanVectors(:,gmmnum),gmmParameters.covMatrices(:,:,gmmnum)); % Evaluate p(x|L=GMM_number)
end
% correct for class labels (for Gauss 1 and 2 the label is 0, 3 is label 1)
pxgivenl(1,:) = 0.5*pxgivenGMM(1,:) + 0.5*pxgivenGMM(2,:); % P(L=0) = P(x in GMM1) + P(x in GMM2)
pxgivenl(2,:) = pxgivenGMM(3,:);

% px = gmmParameters.priors*pxgivenl; % Total probability theorem
% classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N(4))./repmat(px,2,1); % P(L=l|x)
% 
% lossMatrix = ones(2,2)-eye(2); % For min-Perror design, use 0-1 loss
% expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
% [~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% determine threshold test.
LR = pxgivenl(2,:)./pxgivenl(1,:);
[pfa,pd,pmd,~,gamma] = binClassifier(LR,data(4).labels); % Grabs prob of fa, detection, etc.

figure(1);
plot(pfa,pd,'*');
title('ERM Classifier ROC N=10000');
xlabel('P(D=1|L=0;\gamma)'); ylabel('P(D=1|L=1;\gamma)');

% Minimize p(error;gamma) with gamma and plot this gamma on ROC

Perror = pfa.*pL0 + pmd.*pL1; % probability of error
[~,idx] = min(Perror);
gamma_mpe = gamma(idx)

hold on
plot(pfa(idx),pd(idx),'r+','MarkerSize',10,'LineWidth',3);
legend('ROC',['min p(error;\gamma = ' num2str(gamma_mpe) ')=' num2str(Perror(idx))],...
                'Location','southeast');
figure(1);
ax = gca;
ax.FontSize = 18;

figure;
plot(gamma,Perror,'+','MarkerSize',6);
axis([0.5 4 0.16 0.19]);
title(["Probability of error near theoretical minimum"; "for varying threshold values"]);
xlabel("\gamma"); ylabel("P_{error}");
ax = gca;
ax.FontSize = 18;

% REPORT THE ESTIMATE gamma_mpe that minimizes the probability of error
% (best achievable for this distribution)
% Theoretical (0-1 loss mtx) 1.8571
%   Empirical estimation: 1.7909

%% Q1 Part 2. MLE - Approximate class label posterior functions
% use data 1-3 to train MLEs using logistic generalized linear models
% code was adapted (replicated) from binaryClassificationGLM.m
clear all; close all;
load("A2Q1Dataset.mat");

% Training parameters:
epsilon = 1e-3; % stopping criterion threshold/tolerance
alpha = 1e-2; % step size for gradient descent methods
NcVal = [length(find(data(4).labels==0)), length(find(data(4).labels==1))];

%%%------- 20 sample dataset: data(1).x -----------%%%
% do each training set 1-3 in a for loop

for i = 1:3
    disp('Training the logistic-linear model with gradient descent.'),
    % Deterministic (batch) gradient descent
    % Uses all samples in training set for each gradient calculation
    paramsGD.type = 'batch';
    paramsGD.ModelType = 'logisticLinear';
    paramsGD.stepSize = alpha;
    paramsGD.stoppingCriterionThreshold = epsilon;
    paramsGD.minIterCount = 10;
    [wGradDescentLin,zLin] = gradientDescent_binaryCrossEntropy(data(i).x,data(i).labels,paramsGD);

    disp('Training the logistic-quadratic model with gradient descent.'),
    % Deterministic (batch) gradient descent 
    % Uses all samples in training set for each gradient calculation
    paramsGD.type = 'batch';
    paramsGD.ModelType = 'logisticQuadratic';
    paramsGD.stepSize = alpha;
    paramsGD.stoppingCriterionThreshold = epsilon;
    paramsGD.minIterCount = 10;
    [wGradDescentQuad,zQuad] =  gradientDescent_binaryCrossEntropy(data(i).x,data(i).labels,paramsGD);
  
    figure(1); plot_training_data(data(i).labels,[1,3,i],zLin,wGradDescentLin,'L',n); % in is data dimension
    title(['2D Linear GD Training Based on', num2str(N(i)),'Samples']); % N is vector containing number of samples

    figure(2); 
    plot_training_data(data(i).labels, [1,3,i], zQuad,wGradDescentQuad,'Q',n)
    title(['2D Quadaratic GD Training Based on', num2str(N(i)),'Samples']);
 
    % Linear: use validation data(10k points) and make decisions
    % -> validation set is in index 4.
    test_set_L=[ones(1,N(4)); data(4).x];
    decision_L_GD=wGradDescentLin'*test_set_L>=0;

    test_set_Q= [ones(1,N(4)); data(4).x]; % setup linear components
    for r = 1:n
        for c = 1:n
            test_set_Q = [test_set_Q ; data(4).x(r,:).*data(4).x(c,:)];
        end
    end
    decision_Q_GD=wGradDescentQuad'*test_set_Q>=0; 
    
    % VALIDATION PLOTS
%     % Plot decision and boundary line (linear)
%     figure(3); 
%     error_L_GD(i)=plot_classified_data(decision_L_GD,data(4).labels, NcVal,gmmParameters.priors ,... 
%         [1,3,i], test_set_L,wGradDescentLin,'L',n);
%     title(['2D Linear GD Classification Based on' , num2str(N(i)),'Samples']);
%     
%     %Quadratic: plot all decisions and boundary contour 
%     figure(4);
%     error_quad_GD(i)= plot_classified_data(decision_Q_GD, data(4).labels,NcVal,...
%         gmmParameters.priors,[1,3,i],test_set_Q,wGradDescentQuad,'Q',n);
%     title(['2D Quadratic GD Classification Based on', num2str(N(i)),'Samples']);

    P_error_Lin(i) = sum(xor(decision_L_GD,data(4).labels))/N(4); % Sum of mistakes divided by total samples
    % result is about 38.86 percent error down to around 34 percent.
    P_error_Quad(i) = sum(xor(decision_Q_GD,data(4).labels))/N(4); % Sum of mistakes divided by total samples

end

%% Question 2
% setup: get data and save it then comment these out.
clear all; close all;
% [xTrain,yTrain,xValidate,yValidate] = hw2q2(100,1000); % using 100 training samples and 1000 validation
% save("A2Q2Dataset.mat");


%% Question 2 begin
clear all; close all;
load("A2Q2Dataset.mat");

% plot if you want it
% figure; plot3(xTrain(1,:),xTrain(2,:),yTrain,'.'), axis equal,
% xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset');

x = xTrain;
N = size(x,2); %number of data samples
epsilon = 1e-3; % stopping criterion threshold/tolerance
alpha = 1e-6; % step size for gradient descent methods

paramsGD.type = 'batch';
%paramsGD.ModelType = 'logisticLinear';
paramsGD.stepSize = alpha;
paramsGD.stoppingCriterionThreshold = epsilon;
paramsGD.minIterCount = 10;
% [w,z] = gradientDescent_ML_Cubic(x,yTrain,paramsGD);
    % gradient descent totally doesn't work.
[~,z] = generalizedLinearModelCubic(x,randn(11,1));

% can set gradient equal to zero and solve (numerically), but the answer is
% available analytically as 
R = (z*z')./N; q = z*yTrain'./N; % N is size of data samples
wML = inv(R)*q; % should give the same thing as gradient descent.

% Do the validation step with wML
[c,~] = generalizedLinearModelCubic(xValidate,wML);
MSE = mean((yValidate - c).^2); % Mean Square Error

% plot if you want it
figure; plot3(xValidate(1,:),xValidate(2,:),yValidate,'.'), axis equal,
hold on;
plot3(xValidate(1,:),xValidate(2,:),c,'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), 
title('Validation of ML parameters');

%%

clear all; close all;
load("A2Q2Dataset.mat");

% plot if you want it
% figure; plot3(xTrain(1,:),xTrain(2,:),yTrain,'.'), axis equal,
% xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset');

x = xTrain;
N = size(x,2); %number of data samples
sig2 = 1; % data additive noise variance
gamma = logspace(-4,4,1000); % init gamma
% optimization options
options = optimset('MaxFunEvals',10000,'TolFun',1e-3,'TolX',1e-3); % will stop when change in w
           %is less than TolX and change in fxn value is less than 1e-3
% initializations for optimization:
% w0 = 0.1*randn(11,1); % initialize w0
w0 = zeros(11,1); % initialize w0 to zeros (zero mean prior)
[~,z] = generalizedLinearModelCubic(x,w0); % get z
    
% minimize for every gamma
for i = 1:length(gamma)

    % MAP training using fminsearch
    cost = @(w)(1/N)*sum((yTrain-w'*z).^2) + sig2/(N*gamma(i))*(w'*w);
    wMAP = fminsearch(cost,w0,options); % minimization function

    % Do the validation step with wMAP
    [c,~] = generalizedLinearModelCubic(xValidate,wMAP);
    MSE_map(i) = mean((yValidate - c).^2); % Mean Square Error
end

% % plot if you want it
% figure; plot3(xValidate(1,:),xValidate(2,:),yValidate,'.'), axis equal,
% hold on;
% plot3(xValidate(1,:),xValidate(2,:),c,'.'), axis equal,
% xlabel('x1'),ylabel('x2'), zlabel('y'), 
% title('Validation of MAP parameters');

figure;
semilogx(gamma,MSE_map);
title('Mean Square Error MAP Estimation vs. \gamma');
xlabel('\gamma'); ylabel('MSE');
ax = gca;
ax.FontSize = 18;

%% --------------UTILITY FUNCTIONS---------------------

function [Pfp,Ptp,Pmd,Perror,thresholdList] = binClassifier(discriminantScores,labels)
% Class labels should be either 0 or 1.
% sort the discriminantScores (i.e. values... for LRT this is the
% likelihood ratio ...
% Also the Perror is accurate only for equally weighted prior
% probabilities.
[sortedScores,ind] = sort(discriminantScores,'ascend');
% Creates thresholds in between every piece of data
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Pmd(i) = length(find(decisions==0 & labels==1))/length(find(labels==1));
    Pfp(i) = length(find(decisions==1 & labels==0))/length(find(labels==0));
    Ptp(i) = length(find(decisions==1 & labels==1))/length(find(labels==1));
    Perror(i) = sum(decisions~=labels(ind))/length(labels); % Error for P(L=1) = P(L=0) = 0.5
end
end
%
%------------------------------------------------------------------------%
%
function [w,z] = gradientDescent_ML_Cubic(x,labels,gradDescentParameters)
% labels in this case are the yTrain data

N = size(x,2); % total number of samples

% Training weights using gradient descent

    alpha = gradDescentParameters.stepSize;
    epsilon = gradDescentParameters.stoppingCriterionThreshold;
    minIterCount = gradDescentParameters.minIterCount;
    
    %Initialize estimates for weights, cubic will have 11
    w = 0.01*randn(11,1);
    
    [cost,c,gradient,z] = MLCostFunctionCubic(w,x,labels);
    iterCounter = 0;
    %perform gradient descent
    while iterCounter < minIterCount || norm(gradient) > epsilon
        
        w = w - alpha*gradient;
        [cost,c,gradient,z] = MLCostFunctionCubic(w,x,labels);
        iterCounter = iterCounter + 1;
    end
end
%
%------------------------------------------------------------------------%
%
function [cost,c,gradient,z] = MLCostFunctionCubic(w,x,yTrain)
% This is the Maximum likelihood cost function
% Inputs are: (w,x,yTrain)

N = size(x,2); % number of data samples 
% Cost function to be minimized to optimize w
[c,z]=generalizedLinearModelCubic(x,w); % linear model function approximation
cost = (1/N)*sum((yTrain-c).^2);
gradient = -2/N*(z*yTrain') + 2/N*(z*z')*w; % gradient of the cost  function with respect to the wieghts w 

% Outputs are: [cost,c,gradient,z]
end
%
%------------------------------------------------------------------------%
%
function [c,z] = generalizedLinearModelCubic(x,w)
% Approximation of a cubic function of x

N = size(x,2); % determine size of data vectors
n=size(x,1); % determine dimensionality of data vectors

% Data augmentation for 2 dimensional cubic model
z = [ones(1,N);x];
for r = 1:n
    for c = 1:n
        z = [z;x(r,:).*x(c,:)];
        z = [z;x(r,:).*x(c,:).*x(r,:)]; % include the cubic terms
    end
end

c = w'*z; % linear model function
end
