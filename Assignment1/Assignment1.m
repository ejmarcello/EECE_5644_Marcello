%% Assignment 1 script
%
% By Ethan Marcello
%
% TODO: 

%% Question 1 (setup)

% Setup: create 10k samples based on data distribution & keep track of
% class labels.
N = 10000; % 10k total number of samples
n = 4; % dimensionality of the data vector
m0 = -1*ones(n,1); % mean vector for L=0. Length is same as n
m1 = ones(n,1); % mean vector for L=1

C0 = [2 -0.5 0.3 0;      % covariance matrix L=0
      -0.5 1 -0.5 0;
      0.3 -0.5 1 0;
      0 0 0 2];
C1 = [1 0.3 -0.2 0;     % covariance matrix L=1
      0.3 2 0.3 0;
      -0.2 0.3 1 0;
      0 0 0 3];

% class priors
pL0 = 0.7;
pL1 = 0.3;  
  % Using class priors to generate data x
x0 = randGaussian(N*pL0,m0,C0); % L=0 data, Uses randGaussian function
x0(:,:,2) = zeros(n,N*pL0); % Add label to 3rd dimension of the mtx (not 3rd dimension of data)
x1 = randGaussian(N*pL1,m1,C1); % L=1 data, Uses randGaussian function
x1(:,:,2) = ones(n,N*pL1); % Add label to 3rd dimension of the mtx


x = [x0(:,:,:) x1(:,:,:)]; % We know all L=0 data is in x0 and all L=1 data is in x1

% Save this dataset to use for all cases.
%save('Q1Dataset');
% instead I could go ahead and reset the randomizer seed at the top and it
% should still generate the same random data every time. Leave it alone for
% now.

%% Question 1 (Cont'd)
% Part A: ERM classification using the knowledge of true data pdf
clear all; close all;
load('Q1Dataset');

%%% Part A 2. %%%

% class priors
pL0 = 0.7;
pL1 = 0.3;

% use evalGaussianPDF function to return a row of p(x_i) values
pxL0 = evalGaussianPDF(x(:,:,1),m0,C0);
pxL1 = evalGaussianPDF(x(:,:,1),m1,C1);
LR = pxL1./pxL0; % Likelihood ratio for all the samples

% Input discriminatorScore and class Labels (must be 0 or 1)
[Pfp,Ptp,Pmd,~,thresholdList] = binClassifier(LR,x(1,:,2));
% Pfp = prob. false positive; Ptp = prob. true positive; Pmd = prob. missed
% detection.
% Perror is the error with equal weighted priors.

figure(1);
plot(Pfp,Ptp,'*');
title('ERM Classifier ROC Question 1 Part A 2.');
xlabel('P(D=1|L=0;\gamma)'); ylabel('P(D=1|L=1;\gamma)');
%min(pd) % if zero ensures we achieve 0,0 point

%%% Part A 3. %%%
% Minimize p(error;gamma) with gamma and plot this gamma on ROC

Perror = Pfp.*pL0 + Pmd.*pL1; % probability of error
[~,idx] = min(Perror);
gamma_mpe = thresholdList(idx)

hold on
plot(Pfp(idx),Ptp(idx),'r+','MarkerSize',10,'LineWidth',3);
legend('ROC',['min p(error;\gamma = ' num2str(gamma_mpe) ')=' num2str(Perror(idx))],...
                'Location','southeast');
figure(1);
ax = gca;
ax.FontSize = 18;

Perror = Pfp.*pL0 + Pmd.*pL1; % probability of error
figure;
plot(thresholdList,Perror,'+','MarkerSize',6);
axis([1 4 0.02 0.04]);
title(["Probability of error near theoretical minimum"; "for varying threshold values"]);
xlabel("\gamma"); ylabel("P_{error}");
ax = gca;
ax.FontSize = 18;

% REPORT THE ESTIMATE gamma_mpe that minimizes the probability of error
% (best achievable for this distribution)
% Theoretical (0-1 loss mtx) 2.3333
%   Empirical estimation: 3.05

%% PART B: Same as part A but now covariance matricies are diagonal.
% Observe changes:

% Naive Bayesian Covariance assumption is diagonal:
C0 = [2 0 0 0;      % covariance matrix L=0
      0 1 0 0;
      0 0 1 0;
      0 0 0 2];
C1 = [1 0 0 0;     % covariance matrix L=1
      0 2 0 0;
      0 0 1 0;
      0 0 0 3];

%%% Part B 2. %%%

% class priors
pL0 = 0.7;
pL1 = 0.3;

% use evalGaussianPDF function to return a row of p(x_i) values
pxL0 = evalGaussianPDF(x(:,:,1),m0,C0);
pxL1 = evalGaussianPDF(x(:,:,1),m1,C1);
LR = pxL1./pxL0; % Likelihood ratio for all the samples

[pfa,pd,pmd,~,gamma] = binClassifier(LR,x(1,:,2)); % Grabs prob of fa, detection, etc.

figure;
plot(pfa,pd,'*');
title('ERM Naive Bayesian Classifier ROC'); % Question 1 Part B 2.
xlabel('P(D=1|L=0;\gamma)'); ylabel('P(D=1|L=1;\gamma)');
%min(pd) % if zero ensures we achieve 0,0 point

%%% Part B 3. %%%
% Minimize p(error;gamma) with gamma and plot this gamma on ROC

pe = pfa.*pL0 + pmd.*pL1; % probability of error
[~,idx] = min(pe);
gamma_mpe = gamma(idx)

hold on
plot(pfa(idx),pd(idx),'r+','MarkerSize',10,'LineWidth',3);
legend('ROC',['min p(error;\gamma = ' num2str(gamma_mpe) ')' num2str(pe(idx))],...
                'Location','southeast');
ax = gca;
ax.FontSize = 18;

% REPORT THE ESTIMATE gamma_mpe that minimizes the probability of error
% (best achievable for this distribution)
% Theoretical (0-1 loss mtx) 2.3333
%   Empirical estimation: 1.115


%% PART C : Fischer LDA Classifier

%%% Part C 2 %%%
% Assume class labels are KNOWN on each data point.
% First Estimate the parameters with known data class labels:
% Estimate mean vectors and covariance matrices from samples
mu0hat = mean(x0(:,:,1),2); S1hat = cov(x0(:,:,1)');
mu1hat = mean(x1(:,:,1),2); S2hat = cov(x1(:,:,1)');

% Calculate the between/within-class scatter matrices
% ***Assume equal weights on the priors when calculating these:
Sb = (mu0hat-mu1hat)*(mu0hat-mu1hat)';
Sw = S1hat + S2hat;

% Solve for the Fisher LDA projection vector (in w) by finding the largest
% eigenvalue and its vector pair of inv(Sw)*Sb
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector (vector corresponding to largest
                    % eigenvalue.

% Linearly project the data from both categories on to w
y0 = w'*x0(:,:,1);
y1 = w'*x1(:,:,1);
if mean(y1)<=mean(y0) w = -w; end % fixing for negative projection
y = w'*x(:,:,1);
% y is now the discriminator score needed to be thresholded.

[Pfp,Ptp,Pmd,~,gammaLDA] = binClassifier(y,x(:,:,2));

figure;
plot(Pfp,Ptp,'*');
title('ERM Fisher LDA Classifier ROC');
xlabel('P(D=1|L=0;\gamma)'); ylabel('P(D=1|L=1;\gamma)');
%min(pd) % if zero ensures we achieve 0,0 point

%%% Part C 3. %%%
% Minimize p(error;gamma) with gamma and plot this gamma on ROC

Perror = Pfp.*pL0 + Pmd.*pL1; % probability of error
[~,idx] = min(Perror);
gamma_mpe = gammaLDA(idx)

hold on
plot(Pfp(idx),Ptp(idx),'r+','MarkerSize',10,'LineWidth',3);
legend('ROC',['min p(error;\gamma = ' num2str(gamma_mpe) ')=' num2str(Perror(idx))],...
                'Location','southeast');
ax = gca;
ax.FontSize = 18;

%% Question 2 (setup)
clear all; close all;

% Setup: create 10k samples based on data distribution & keep track of
% class labels.
N = 10000; % 10k total number of samples
n = 3; % dimensionality of the data vector
sep = 2; % Data mean separation parameter
L = 3;
gmmParameters.priors = [0.3 0.3 0.4];
% Construct data means by placing them on the corners of a cube
m1 = zeros(n,1); % mean vector for L=0. Length is same as n
m2 = [sep 0 0]'; % mean vector for L=2
m3 = [0 sep 0]'; % mean vector for L=3
m4 = [0 0 sep]'; % mean vector for L=4
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
C1 = 0.2.*(eye(3)+s1.*A1)*(eye(3)+s1.*A1)';
C2 = 0.2.*(eye(3)+s2.*A2)*(eye(3)+s2.*A2)';
C3 = 0.2.*(eye(3)+s3.*A3)*(eye(3)+s3.*A3)';
C4 = 0.2.*(eye(3)+s4.*A4)*(eye(3)+s4.*A4)';
gmmParameters.covMatrices(:,:,1) = C1;
gmmParameters.covMatrices(:,:,2) = C2;
gmmParameters.covMatrices(:,:,3) = C3;
gmmParameters.covMatrices(:,:,4) = C4;

  % Using class priors to generate data x
x1 = randGaussian(N*0.3,m1,C1); % L=1 data, Uses randGaussian function
x1(:,:,2) = ones(n,N*0.3); % Add label to 3rd dimension of mtx
x2 = randGaussian(N*0.3,m2,C2); % L=2 data, Uses randGaussian function
x2(:,:,2) = 2.*ones(n,N*0.3); % Add label to 3rd dimension
x3 = randGaussian(N*0.2,m3,C3); % L=3 data, first gaussian model
x3(:,:,2) = 3.*ones(n,N*0.2); % Add label to 3rd dimension
x4 = randGaussian(N*0.2,m4,C4); % L=3 data, second gaussian model
x4(:,:,2) = 3.*ones(n,N*0.2); % Add label to 3rd dimension

x = [x1(:,:,:) x2(:,:,:) x3(:,:,:) x4(:,:,:)]; 
% We know all L=0 data is in x0 and all L=1 data is in x1, etc.
labels = x(1,:,2);

% Save this dataset to use for all cases.
%save('Q2Dataset');

% Plot data in 3D to see level of overlap
% plot3(x1(1,:)',x1(2,:)',x1(3,:)','r+')
% hold on
% plot3(x2(1,:)',x2(2,:)',x2(3,:)','go')
% plot3(x3(1,:)',x3(2,:)',x3(3,:)','c*')
% plot3(x4(1,:)',x4(2,:)',x4(3,:)','c*')

%% Question 2 Part A 1-3
clear all; close all;
load('Q2Dataset.mat');

% begin Question 2:
%%% Part A.1 %%%

% gmmnum is the number of the gaussian in the mixture model.
for gmmnum = 1:length(gmmParameters.meanVectors(1,:))
    pxgivenGMM(gmmnum,:) = evalGaussianPDF(x(:,:,1),gmmParameters.meanVectors(:,gmmnum),gmmParameters.covMatrices(:,:,gmmnum)); % Evaluate p(x|L=GMM_number)
end
% correct for class labels (for Gauss 3 and 4 the label is 3)
pxgivenl = pxgivenGMM;
pxgivenl(3,:) = 0.5*pxgivenl(3,:) + 0.5*pxgivenl(4,:); % P(L=3) = P(x in GMM3) + P(x in GMM4)
pxgivenl = pxgivenl(1:3,:);

px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,L,1); % P(L=l|x)

%%% Question 2.A.2 %%%

lossMatrix = ones(L,L)-eye(L); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% Find confusion matrix using function confusionMatrix.m
% labels was already defined in the dataset. Also saved under x(:,:,2)
[confusionMtx,classPriors,expectedRisk] = confusionMatrix(labels,decisions,lossMatrix);
confusionMtx

%%% Question 2.A.3 %%%

pA3_scatter = ERMscatter(x,labels,decisions); % wrote plot function to declutter code


%% Question 2.B

loss10 = [0 1 10; 1 0 10; 1 1 0]; % Higher loss for incorrect decision on L=3
loss100 = [0 1 100; 1 0 100; 1 1 0];

% Find new Expected Risks & subsequent minimum risk decisions with new loss matricies
expectedRisks10 = loss10*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
expectedRisks100 = loss100*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions10] = min(expectedRisks10,[],1); % Minimum expected risk decision 10x loss
[~,decisions100] = min(expectedRisks100,[],1); % Minimum expected risk decision with 100x loss

% Start analysis using confusion matricies
[confusionMtx10,classPriors,expectedRisk] = confusionMatrix(labels,decisions10,loss10);
confusionMtx10
[confusionMtx100,classPriors,expectedRisk] = confusionMatrix(labels,decisions100,loss100);
confusionMtx100

% From the results of the confusion matricies, it can be seen that my
% increasing the loss coefficient for wrong decisions when the true class
% label is L=3 causes a significant decrease in the percentage of correct
% decisions made for the L=1 class, with most of these errors being
% misidentified as class 3. The correct detections for class L=2 are also
% effected, although only slightly. The most accuracy lost is just over
% 2.5% with the error leaning towards decisions for L=1 at first and
% growing to L=3 at the extreme.

pB10_scatter = ERMscatter(x,labels,decisions10);
title('Classification scatter plot with 10x loss')
pB100_scatter = ERMscatter(x,labels,decisions100);
title('Classification scatter plot with 100x loss')

%% QUESTION 3: Setup - data construction
clear all; close all;
% import wine quality dataset
% https://archive.ics.uci.edu/ml/datasets/Wine+Quality
winedata = readmatrix('../Datasets/UCI Wine Quality/winequality-white.csv')';
% imports rows 1-11 as data dimensions. row 12 is classification label
winelabels = winedata(12,:); % possible class labels are [0,10] inclusive.
winedata = winedata(1:11,:); % 0,1,2,10 are not present in the data. Set priors to 0

% save('WineDataset.mat','winedata','winelabels');

% import data from HAR dataset
% https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
HARdata = readmatrix("../Datasets/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt")';
    % read in 561 dimensional data (561 features) - 7352 samples. (training
    % data)
HARlabels = readmatrix("../Datasets/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt")';
    % class labels are 1-6

% save('UCI_HAR_Dataset.mat','HARdata','HARlabels');

%% Wine dataset calculations
clear all; close all;
% start with the wine dataset
load('WineDataset.mat');

N = size(winedata,2); % number of samples

% GOAL: Implement minimum-probability-of-error classifier

% Automate collection of data into classes based on class labels.
[winecldata,uql] = extractClassData(winedata,winelabels);
% access class sample data by winecldata(uql(label_num)).samples

% Estimate class priors from samples.
    for i = uql
        priors(i) = size(winecldata(i).samples,2)/size(winelabels,2);
    end

% ASSUME class-conditional pdfs are Gaussian. Estimate means and Covariances
[muhat_w, Sigmahat_w] = meanCovEstimates(winecldata,uql); % subscript w stands for "wine"
% class labels are correct, but class L=0 is not included (which is okay
% becuase there is no data with L=0.

% Regularize covariance matricies: "CRegularized = CSampleAverage+λI where λ > 0 is a small regularization parameter that ensures
% the regularized covariance matrix CRegularized has all eigenvalues larger
% than this parameter."
% The regularization term is used to "smooth" out the Gaussians. Higher
% regularization smooths out more.
%
lambda = 0.1; % regularization term
    for i = 1:size(Sigmahat_w,3)
        % Obtain regularized covariances to ensure sample estimates are not
        % ill-posed matricies.
        Sigmahat_wR(:,:,i) = Sigmahat_w(:,:,i) + lambda*eye(size(Sigmahat_w(:,:,i)));
    end
    
gmmParameters.priors = priors; % row vector ***
gmmParameters.meanVectors = muhat_w;
gmmParameters.covMatrices = Sigmahat_wR;

% Calculate class-conditional PDF
% cl is the class label
for cl = 1:length(gmmParameters.meanVectors(1,:))
    pxgivenl(cl,:) = evalGaussianPDF(winedata,gmmParameters.meanVectors(:,cl),gmmParameters.covMatrices(:,:,cl)); % Evaluate p(x|L=cl)
end
% now have P(x|L=cl)

NL = size(pxgivenl,1); % number of class labels
px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,NL,1); % P(L=l|x)

%%% Calculate loss matrix and find minimum risk decisions

lossMatrix = ones(NL,NL)-eye(NL); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,winedecisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% Could not use function confusionMatrix.m, due to the fact that some data
% classes are not represented in the data. Code was altered and new
% function was used.
[confusionMatrix_w,classPriors_w,expectedRisk_w,Perror_w] = confusionMatrixx(winelabels,winedecisions,lossMatrix);

%Print these to command window.
Perror_w
P_correct_w = 1-Perror_w
max(gmmParameters.priors)

%% Run PCA dimension reduction for wine dataset 
% and show the first two principle components.

Lpca = 4; % Implement PCA dimension reduction for class label L=6 
Nclsamp = size(winecldata(Lpca).samples,2); % number of samples for the chosen class


% Get eigenvalues/vectors for Sigmaest
[Q,D] = eig(gmmParameters.covMatrices(:,:,Lpca)); % D is diagonal mtx, Q are corresponding eigvects as col vectors

lambdas = diag(D);
[lambdas,Is] = sort(lambdas,'descend');
Q = Q(:,Is);
D = diag(lambdas);

% want m-dimensional samples m<n
m = 2; % get first two principle components
Qm = Q(:,1:m); % take first m number of eigenvectors to create m dimensional data

xzm = (winecldata(Lpca).samples - repmat(gmmParameters.meanVectors(:,Lpca),1,Nclsamp));
y = Qm'*xzm; % dimension-reduced data!

figure;
plot(y(1,:),y(2,:),'+','MarkerSize',8);
title("PCA on white wine data")
legend("Class Label L=" + num2str(Lpca));
xlabel("1^{st} Principle Component");
ylabel("2^{nd} Principle Component");
axis equal
ax = gca;
ax.FontSize = 16;

%% Run PCA on all Wine data and show separation of classes
% Only use 2D scatterplot because third PC doesn't add any amplifying
% information...

muest = mean(winedata,2);
Sigmaest = cov(winedata');

lambda = 0.1;
Sigmaest_R = Sigmaest + lambda.*eye(length(muest)); % Regularize covariance estimate
% Get eigenvalues/vectors for Sigmaest_R
[Q,D] = eig(Sigmaest_R); % D is diagonal mtx, Q are corresponding eigvects as col vectors

lambdas = diag(D);
[lambdas,Is] = sort(lambdas,'descend');
Q = Q(:,Is);
D = diag(lambdas);

% want m-dimensional samples m<n
m = 3; % get first three principle components
Qm = Q(:,1:m); % take first m number of eigenvectors to create m dimensional data

xzm = (winedata - repmat(muest,1,N));
y = Qm'*xzm; % dimension-reduced data!

figure;
for cl = 1:max(winelabels)
    idxcl = find(winelabels==cl);
    if isempty(idxcl)
        plot(0,0,'+');
        hold on
    else
        plot(y(1,idxcl),y(2,idxcl),'+','MarkerSize',8);
        hold on
    end
end
title("PCA on wine data")
xlabel("1^{st} PC");
ylabel("2^{nd} PC"); %zlabel("3^{rd} PC");
legend('L=1','L=2','L=3','L=4','L=5','L=6','L=7','L=8','L=9')
axis equal
ax = gca;
ax.FontSize = 16;

%% HAR dataset calculations
clear all; close all;
% start with the wine dataset
load('UCI_HAR_Dataset.mat');

N = size(HARdata,2); % number of samples

% GOAL: Implement minimum-probability-of-error classifier

% Automate collection of data into classes based on class labels.
[HARcldata,uql] = extractClassData(HARdata,HARlabels);
% access class sample data by winecldata(uql(label_num)).samples


% Estimate class priors from samples.
    for i = uql
        priors(i) = size(HARcldata(i).samples,2)/size(HARlabels,2);
    end

% ASSUME class-conditional pdfs are Gaussian. Estimate means and Covariances
[muhat_H, Sigmahat_H] = meanCovEstimates(HARcldata,uql); % subscript w stands for "wine"


% Regularize covariance matricies as in wine data
lambda = 0.5; % regularization term, had to increase to avoid ill-posed matrices
    for i = 1:size(Sigmahat_H,3)
        % Obtain regularized covariances to ensure sample estimates are not
        % ill-posed matricies.
        Sigmahat_HR(:,:,i) = Sigmahat_H(:,:,i) + lambda*eye(size(Sigmahat_H(:,:,i)));
    end
    
gmmParameters.priors = priors; % row vector ***
gmmParameters.meanVectors = muhat_H;
gmmParameters.covMatrices = Sigmahat_HR;

% Calculate class-conditional PDF
% cl is the class label
for cl = 1:length(gmmParameters.meanVectors(1,:))
    pxgivenl(cl,:) = evalGaussianPDF(HARdata,gmmParameters.meanVectors(:,cl),gmmParameters.covMatrices(:,:,cl)); % Evaluate p(x|L=cl)
end
% now have P(x|L=cl)


NL = size(pxgivenl,1); % number of class labels
px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,NL,1); % P(L=l|x)

%%% Calculate loss matrix and find minimum risk decisions

lossMatrix = ones(NL,NL)-eye(NL); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,HARdecisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% Used function confusionMatrix.m
[confusionMatrix_H,classPriors_H,expectedRisk_H] = confusionMatrix(HARlabels,HARdecisions,lossMatrix);

P_correct_H = sum(diag(confusionMatrix_H).*gmmParameters.priors')
Perror_H = 1-P_correct_H
max(gmmParameters.priors)

%% Run PCA on HAR for a single class label
% objective is to show the first two principle components.

Lpca = 6; % Implement PCA dimension reduction for class label L=6 (since we always correctly
%classified it, so want to see if gaussian assumption was actaully good)
Nclsamp = size(HARcldata(Lpca).samples,2); % number of samples for the chosen class


% Get eigenvalues/vectors for Sigmaest
[Q,D] = eig(gmmParameters.covMatrices(:,:,Lpca)); % D is diagonal mtx, Q are corresponding eigvects as col vectors

lambdas = diag(D);
[lambdas,Is] = sort(lambdas,'descend');
Q = Q(:,Is);
D = diag(lambdas);

% want m-dimensional samples m<n
m = 2; % get first two principle components
Qm = Q(:,1:m); % take first m number of eigenvectors to create m dimensional data

xzm = (HARcldata(Lpca).samples - repmat(gmmParameters.meanVectors(:,Lpca),1,Nclsamp));
y = Qm'*xzm; % dimension-reduced data!

figure;
plot(y(1,:),y(2,:),'+','MarkerSize',8);
title("PCA on HAR data")
legend("Class Label L=" + num2str(Lpca));
xlabel("1^{st} Principle Component");
ylabel("2^{nd} Principle Component");
axis equal
ax = gca;
ax.FontSize = 16;

%% Run PCA on all HAR data and show separation of classes
% use 3D scatterplot.

muest = mean(HARdata,2);
Sigmaest = cov(HARdata');

lambda = 0.5;
Sigmaest_R = Sigmaest + lambda.*eye(length(muest)); % Regularize covariance estimate
% Get eigenvalues/vectors for Sigmaest_R
[Q,D] = eig(Sigmaest_R); % D is diagonal mtx, Q are corresponding eigvects as col vectors

lambdas = diag(D);
[lambdas,Is] = sort(lambdas,'descend');
Q = Q(:,Is);
D = diag(lambdas);

% want m-dimensional samples m<n
m = 3; % get first three principle components
Qm = Q(:,1:m); % take first m number of eigenvectors to create m dimensional data

xzm = (HARdata - repmat(muest,1,N));
y = Qm'*xzm; % dimension-reduced data!

figure;
symbs = {'+','r+','g+','k+','c+','y+'};
for cl = unique(HARlabels)
    idxcl = find(HARlabels==cl);
    plot3(y(1,idxcl),y(2,idxcl),y(3,idxcl),symbs{cl},'MarkerSize',8);
    hold on
end
title("PCA on HAR data")
xlabel("1^{st} PC");
ylabel("2^{nd} PC"); zlabel("3^{rd} PC");
legend('L=1','L=2','L=3','L=4','L=5','L=6')
axis equal
ax = gca;
ax.FontSize = 16;

%% Utility functions

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
% mu must be a column vector.
%randGaussian.m Generate N samples of n dimensional data with Gaussian
%distribution ~(mu,Sigma)
%   N is number of samples
%   n is dimension of the data (taken from mu)
%   mu is the mean vector (nx1)
%   Sigma is the covariance matrix (nxn)
%
%   The outut is stored in the variable x.
% Assume Sigma is symmetric, positive semidefinite

if size(mu,2) ~= 1 && size(mu,1) == 1
    mu = mu';
elseif size(mu,2)>1 && size(mu,1)>1
    x = -1;
    return
end

n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2); % Could also get this from eigendecomposition
x = A*z + repmat(mu,1,N); % repmat makes N copies of the mu vector into an array.

end
                            %%%%%%%%%%
%-------------------------------------------------------------------------------%
                            %%%%%%%%%%
                            
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
                            %%%%%%%%%%
%-------------------------------------------------------------------------------%
                            %%%%%%%%%%

function [databyclass,uql] = extractClassData(data,labels)
% EXTRACTCLASSDATA.M Extracts class data using the dataset and the class
% labels. 
% -Input is expected to have samples along the columns. Data dimensionality
% (features) should be on the increasing rows.
%The output is an indexable structure element of sample data
% indexed by the unique labels uql which is also an output.
uql = unique(labels);
% Need to write something in here to account for class labels of 0...
    for i = 1:length(uql)
        idx = find(labels==uql(i));
        databyclass(uql(i)).samples = data(:,idx);
    end
    
end


function [muhat,Sigmahat] = meanCovEstimates(databyclass,uql)
%%% MEANCOVESTIMATES.m finds sample mean and covaraiance for data from each
%%% class label.
% Inputs should be outputs from extractClassData.m

    for i = uql
        muhat(:,i) = mean(databyclass(i).samples,2); 
        Sigmahat(:,:,i) = cov(databyclass(i).samples');
    end

end

function [confusionMatrix,classPriors,expectedRisk,pError] = confusionMatrixx(labels,decisions,lossMatrix)
% assumes labels are in {1,...,L} and decisions are in {1,...,D}

L = max(unique(labels)); % changed these to max value.
D = L; % confusion matrix should be square
confusionMatrix = zeros(D,L);
for l = 1:L
    Nl = length(find(labels==l));
    for d = 1:D
        Ndl = length(find(labels==l & decisions==d));
        if Nl == 0 % Assume if no labels then no decisions were made there either.
           confusionMatrix(d,l) = 0;
        else
           confusionMatrix(d,l) = Ndl/Nl;
        end
    end
    classPriors(l,1) = Nl/length(labels); % class prior for label l
end
if L==D
    pCorrect = sum(diag(confusionMatrix).*classPriors);
    pError = 1-pCorrect;
end
expectedRisk = sum(sum(lossMatrix.*confusionMatrix.*repmat(classPriors,1,L),2),1);
end

function [confusionMatrix,classPriors,expectedRisk] = confusionMatrix(labels,decisions,lossMatrix)
% assumes labels are in {1,...,L} and decisions are in {1,...,D}

arguments
    labels; decisions; lossMatrix = ones(length(unique(labels)),...
            length(unique(labels)))-eye(length(unique(labels)));
end

L = length(unique(labels));
D = length(unique(decisions));
confusionMatrix = zeros(D,L);
for l = 1:L
    Nl = length(find(labels==l));
    for d = 1:D
        Ndl = length(find(labels==l & decisions==d));
        confusionMatrix(d,l) = Ndl/Nl;
    end
    classPriors(l,1) = Nl/length(labels); % class prior for label l
end
if L==D
    pCorrect = sum(diag(confusionMatrix).*classPriors);
    pError = 1-pCorrect;
end
expectedRisk = sum(sum(lossMatrix.*confusionMatrix.*repmat(classPriors,1,L),2),1);
end

function px = evalGaussianPDF(x,mu,Sigma)
% x should have n-dimensional N vectors in columns
    n = size(x,1); % data vectors have n-dimensions
    N = size(x,2); % there are N vector-valued samples
    C = (2*pi)^(-n/2)*det(Sigma)^(-1/2); % normalization constant
    a = x-repmat(mu,1,N); b = inv(Sigma)*a;
    % a,b are preparatory random variables, in an attempt to avoid a for loop
    px = C*exp(-0.5*sum(a.*b,1)); % px is a row vector that contains p(x_i) values
end

function [h] = ERMscatter(x,labels,decisions)
%ermScatter.m - 3D Scatterplot of Expected Risk Minimization
%   Written for assignment 1 to declutter code.
%   Strictly for 3 class labels with D = L = {1,2,3};
%   Output is a handle to the figure.

% Find x-data indicies of decisions
idcG1 = find(labels==1 & decisions==1); % "Indicies for correct decision label = 1"
idcG2 = find(labels==2 & decisions==2);
idcG3 = find(labels==3 & decisions==3);
idcR1 = find(labels==1 & ~(decisions==1)); % Indicies for incorrect decision label=1
idcR2 = find(labels==2 & ~(decisions==2));
idcR3 = find(labels==3 & ~(decisions==3));

%Plot data in 3D scatter plots
h = figure;
plot3(x(1,idcG1)',x(2,idcG1)',x(3,idcG1)','g+')
hold on
plot3(x(1,idcG2)',x(2,idcG2)',x(3,idcG2)','go')
plot3(x(1,idcG3)',x(2,idcG3)',x(3,idcG3)','g*')
% incorrect decisions
plot3(x(1,idcR1)',x(2,idcR1)',x(3,idcR1)','r+')
plot3(x(1,idcR2)',x(2,idcR2)',x(3,idcR2)','ro')
plot3(x(1,idcR3)',x(2,idcR3)',x(3,idcR3)','r*')
title("Classification scatter plot");
legend('D=1 & L=1','D=2 & L=2','D=3 & L=3','D~=1 & L=1',...
        'D~=2 & L=2','D~=3 & L=3','Location','southeast');
xlabel('x1'); ylabel('x2'); zlabel('x3');
end



