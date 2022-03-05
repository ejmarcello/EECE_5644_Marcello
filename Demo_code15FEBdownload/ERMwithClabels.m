% Expected risk minimization with C classes
clear all, close all,

C = 4;
N = 100; % Number of samples
n = 2; % Data dimensionality (must be 2 for plots to work)
gmmParameters.priors = ones(1,C)/C; % uniform priors
gmmParameters.meanVectors = 3*1.25*n*C*(rand(n,C)); % arbitrary mean vectors
for l = 1:C
    A = 5*eye(n)+0.2*randn(n,n);
    gmmParameters.covMatrices(:,:,l) = A'*A; % arbitrary covariance matrices
end

% Generate data from specified pdf
[x,labels] = generateDataFromGMM(N,gmmParameters); % Generate data
for l = 1:C
    Nclass(l,1) = length(find(labels==l));
end

% Shared computation for both parts
for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(x,gmmParameters.meanVectors(:,l),gmmParameters.covMatrices(:,:,l)); % Evaluate p(x|L=l)
end
%pxgivenGMM = w1*evalGaussianPDF(x,m1,C1)+w2*evalGaussianPDF(x,m2,C2);
px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1); % P(L=l|x)

lossMatrix = ones(C,C)-eye(C); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% The matrix computed here is P(D=d|L=l,x)
% It is called the confusion matrix if decisions are class labels.
mShapes = 'ox+*.'; % Accomodates up to C=5
mColors = 'rgbmy';
figure(1), clf,
for d = 1:C % each decision option
    for l = 1:C % each class label
        ind_dl = find(decisions==d & labels==l);
        ConfusionMatrix(d,l) = length(ind_dl)/length(find(labels==l));
        if n == 2
            plot(x(1,ind_dl),x(2,ind_dl),strcat(mShapes(l),mColors(d))), hold on, axis equal,
        end
    end
end
ConfusionMatrix,
    
