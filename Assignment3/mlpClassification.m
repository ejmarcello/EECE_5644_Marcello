%% mlpClassification.m
% The actual MLP classifier doesn't seem to work very well and also takes a
% long time to train. Traded this in for Deep Learning Toolbox and custom
% built my MLP in there.

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
x = dataset(3).x; % choose dataset D (xxxx samples)
labels = dataset(3).labels;
N = N(3); % number of data samples
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

%% Creating the MLP to estimate the class posterior probabilities
% First get the test data
XTest = dataset(7).x'; % these are both transposed to work with deep learning tools
YTest = dataset(7).labels';

% Continues straight from "(benchmark)" section

% Only one hidden layer, one output layer
% Size of MLP
p = 8; n = 3; 
% size of output... equal to number of classes
C = 4;

% START: data.X is data and data.d is true class posteriors
% May want to use class labels via "one-hot" vector format instead... i.e. [0 0 1 0]'
% corresponds to class = 3 or 100 percent posterior prob in L=3 and 0
% elsewhere.
data.X = x; %data.d = classPosteriors; % +1e-2*randn(C,N); % can add some random noise on top
data.d = zeros(n,N); 
for i = 1:N
    data.d(labels(i),i) = 1;
end % now data.d is one-hot vectors
% Randomly initialize parameter estimates for C outputs. Results of MLP will be
% posterior probabilities (using softmax @ the output layer)
% just replace C with 1 to get single output again.
theta.b2 = 0.5*rand(C,1); theta.W2 = 0.5*rand(C,p); theta.b1 = 0.5*rand(p,1); theta.W1 = 0.5*rand(p,n);

% Initialize model to the vicinity of the true MLP (wish I could do this)
%jitter = 5e-1; theta.b2 = ttrue.b2+jitter*randn; theta.W2 = ttrue.W2+jitter*randn(1,p); theta.b1 = ttrue.b1+jitter*randn(p,1); theta.W1 = ttrue.W1+jitter*rand(p,n);

%%%%%---SOLUTION TRY 1---%%%%%
% Train the MLP with SGD starting from specified initial weights
theta = trainmlp(data,theta);


%% Functions for MLP

%%%
% Below, I am assuming that at each iteration we use a single-sample based
% stochastic gradient; will generalize to larger batches later...
function theta = trainmlp(data,theta)
[n,N] = size(data.X);
E = 2; % number of epochs(must be a positive integer)
T = E*N;
t = 0; 
%i = randi(N,1,T); % sample with replacement
i = []; for k = 1:E, i = [i,randperm(N)]; end %Follow random ordering in each epoch
while t <= T
    t = t + 1;
%     eta = 1e-1/(1+1e-3*t);
    eta = 1e-2/(1+1e-3*t); % reduce step size to asymptotically eliminate residual jitter due to stochastic updates
    xt = data.X(:,i(t)); dt = data.d(1,i(t)); % choose a sample randomly
    [e2t,ge2t] = sqerr(xt,dt,theta,[1,1]);
    figure(1), 
    subplot(1,2,1), plot(t,e2t,'.'), xlim([0,T]), xlabel('Iterations'), ylabel('Instantaneous Squared Error'), hold on, drawnow,
    subplot(1,2,2), semilogy(t,e2t,'.'), xlim([0,T]), xlabel('Iterations'), ylabel('Instantaneous Squared Error'), hold on, drawnow,
    theta.b2 = theta.b2 - eta*ge2t.b2;
    theta.W2 = theta.W2 - eta*ge2t.W2;
    theta.b1 = theta.b1 - eta*ge2t.b1;
    theta.W1 = theta.W1 - eta*ge2t.W1;
end
end

%%%
% Below, I am assuming that at each iteration we use a single-sample based
% stochastic gradient; will generalize to larger batches later...
function [e2,ge2] = sqerr(x,d,theta,flag)
if flag(1) | flag(2)
	[y,gy] = mlp(x,theta,flag);
    e = (d - y);
    if flag(1)
        e2 = e.^2;
    end
    if flag(2)
        ge2.b2 = -2*norm(e,2)*gy.b2; % chain rule (d-y)^2 -> 2*e*(-gy.b2)
        ge2.W2 = -2*norm(e,2)*gy.W2';
        ge2.b1 = -2*norm(e,2)*gy.b1;
        ge2.W1 = -2*norm(e,2)*gy.W1;
    else
        ge2 = NaN;
    end
end
end

%%%
function [H,gy] = mlp(x,theta,flag)
% Single hidden layer and one output layer MLP
% flag argument determines if function should fetch the gradient as well.
% altered to include softmax at the output to model class posteriors (H)
[n,N] = size(x); m = length(theta.b1); P = m; % P is number of perceptrons
if flag(1) | flag(2)
    % send data through the MLP
	l = theta.W1*x + theta.b1*ones(1,N);   % first linear layer, result PxN
	[f,gf] = sigmoid(l);  % first nonlinear layer, result [PxN,PxN]
	y = theta.W2*f + theta.b2*ones(1,N);    % output layer, result 1xN
	if flag(2)
        % calculate gradients
        gy.b2 = ones(1,N); % This is partial deriv. w.r.t b2 (the output bias)
        gy.W2 = f; % already has N columns in it
%         gy.b1 = repmat(gy.W2,1,N).*gf; % don't think I neet repmat here. Also chain rule derivative
        gy.b1 = gy.W2.*gf;
        gy.W1 = zeros(m,n,N);
        for j = 1:N % bad for loop that needs to be eliminated
            gy.W1(:,:,j) = gy.b1(:,j)*x(:,j)'; % chain rule derivative
        end
        % Alternative Outer Product Calculation for gy.W1 ... 
%         outer = kron(gy.b1,x); % kronecker tensor product of PxN and nxN sized matricies
%         res = outer(:,1:N+1:N*N); % just take relevant columns to get a P*nxN matrix
%         for j = 1:N
%             gy.W1(:,:,j) = vec2mtx(res(:,j),P,n); % turn the P*nxN to a PxnxN
%                 %  uses another for loop inside, runs array assignment P times.
%         end
    else
        gy = NaN;
    end
    H = exp(y)./sum(exp(y),1); % softmax nonlinearity for second/last layer
    % Activate the softmax function to make this MLP a model for class posteriors
end
end

%%%
function [s,gs] = sigmoid(ksi)
s = 1./(1+exp(-ksi));
gs = s.*(1-s); % derivative
end
% use this instead of sigmoid for activation function
function [s,gs] = softplus(x)
s = log(1 + exp(x));
gs = 1./(1 + exp(-x)); % this is the same thing as the sigmoid
end

function mtx = vec2mtx(vec,P,n)
% Takes in vertical vector vec 1xP*n and turns it into a Pxn matrix
mtx = zeros(P,n);
for c = 1:n:P*n-n+1
    mtx((c-1)/n+1,:) = vec(c:c+n-1)';
end
end

%% Creating an MLP with P perceptrons
% 
% data.x = x;
% data.labels = labels;
% n = size(data.x,1);
% P = 10; % number of perceptrons for the hidden layer could maybe be a vector
%             % to represent multiple layers... [10 5 4]
% % Randomly initialize parameter estimates
% theta.b2 = rand;  % bias term
% theta.w2 = rand(1,P); 
% theta.b1 = rand(P,1); % bias term
% theta.w1 = rand(P,n);
% 
% function out = mlp(data,theta,P)
% % MLP.m is a function that creates a Multi Layer Perceptron network
% %   also known as a feedforward neural network.
% %   Inputs:
% %     data - The input data
% %     theta - The input parameters as a structure containing fields
% %       theta.w and theta.b
% %     P = Pnum;
%     
% end



