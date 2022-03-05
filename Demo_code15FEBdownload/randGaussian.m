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