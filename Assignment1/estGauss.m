function [muest,Sigmaest] = estGauss(x)
%estGauss.m Estimate Mean and Covariance of generated Data.
% dimension should be on rows and data samples on columns. i.e. x is nxN
% where n is dimension of the data and N is number of samples.

N = size(x,2);
n = size(x,1);

muest = 1/N*sum(x,2); % sums the data (across columns) and divides by number of samples

% data samples minus muest
xrmbias = x - repmat(muest,1,N);
% create 3D storage to store Sigmas
Sigmas = zeros(n,n,N);
for i = 1:N
    Sigmas(:,:,i) = xrmbias(:,i)*xrmbias(:,i)';
end
Sigmaest = 1/(N-1)*sum(Sigmas,3);

end

