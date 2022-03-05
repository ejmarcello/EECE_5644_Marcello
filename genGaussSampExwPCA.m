%% genGaussSamplesEx.m (Example for usage)
% Generate Gaussian samples w/ desired covariance matrix and mean using
% randGaussian.m function from class code

% PCA included as a bonus.

N = 100; % number of samples
n = 2; % dimensionality of the data vector
    % (Not needed since we can extract dimensionality from the mu vector)
mu = rand(n,1).*5 + 0.5*ones(n,1); % mean vector. Length is same as n
A = rand(n,n);
% To make Sigma better:
%Sigma = (I+sA)(I+sA)' with 0<s<<1
Sigma = A*A'; % covariance matrix, size nxn symmetric positive semidefinite.

x = randGaussian(N,mu,Sigma);
if n == 2 % visualize 2D data
    plot(x(1,:),x(2,:),'*'); hold on
%    plot(mu(1,:),mu(2,:),'r+');
    axis equal;
    title("Generated Gaussian");
    xlabel('x1'); ylabel('x2');
end

%% Estimate Mean and Covariance of generated Data.
%%% This is now a function called estGauss.m %%%

muest = 1/N*sum(x,2); % sums the data (across columns) and divides by number of samples

% data samples minus muest
xnormal = x - repmat(muest,1,N);
Sigmas = zeros(n,n,N);
for i = 1:N
    Sigmas(:,:,i) = xnormal(:,i)*xnormal(:,i)';
end
Sigmaest = 1/(N-1)*sum(Sigmas,3);

%% Determine Suitable z with dist ~Normal(0,I)
b = muest;
A = Sigmaest^(0.5); % this fact was discovered via going from z to x with
                    % desired distribution i.e. x = A*z + b
W = inv(A);

z = W*(x - repmat(b,1,N)); % 2x2 times 2xN yields 2xN where n=2

[muz,Sigmaz] = estGauss(z); % fxn that estimates gaussian parameters of sample data

%% Implement PCA dimension reduction

% Get eigenvalues/vectors for Sigmaest
[Q,D] = eig(Sigmaest); % D is diagonal mtx, Q are corresponding eigvects as col vectors

lambdas = diag(D);
[lambdas,Is] = sort(lambdas,'descend');
Q = Q(:,Is);
D = diag(lambdas);

% want m-dimensional samples m<n
m = n - 1; % as an example
Qm = Q(:,1:m); % take first m number of eigenvectors to create m dimensional data

xzm = (x - repmat(muest,1,N));
y = Qm'*xzm; % dimension-reduced data!

% Now estimate x from y:
xest = Qm*y + repmat(muest,1,N); % gotta add back the mean estimate

% Plot to compare to x
if n == 2 % visualize 2D data
    plot(xest(1,:),xest(2,:),'o');
    axis equal;
    title("Reconstruction of data after PCA dimension reduction");
    legend('Original','Reconstructed');
end
