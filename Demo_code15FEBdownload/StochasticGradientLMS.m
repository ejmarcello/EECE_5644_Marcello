clear all, close all,

n = 2; N = 1000;
wtrue = 12+rand(n,1),
A = randn(n,n); x = A*randn(n,N);
y = wtrue'*x+1e-3*randn(1,N);

hGrid = linspace(-2,27,101);
vGrid = linspace(-2,27,99);
[h,v] = meshgrid(hGrid,vGrid);
for HG = 1:101
    for VG = 1:99
        mseGrid(VG,HG) = evalMSE([hGrid(HG);vGrid(VG)],x,y);
    end
end
figure(1), subplot(1,2,1),
contour(hGrid,vGrid,mseGrid); axis equal, hold on, 

w(:,1) = zeros(n,1); 
alpha = 5e-3;
T = 1000; ind = zeros(1,T); e = zeros(1,T);
for k = 1:T
    ind(1,k) = randi([1,N],1); % pick a sample randomly from the training set
    e(1,k) = (y(ind(k))-w(:,k)'*x(:,ind(k)));
    figure(1), subplot(1,2,2),semilogy(k,e(1,k)^2,'.'), hold on, drawnow,
    figure(1), subplot(1,2,1),plot(w(1,k),w(2,k),'.'), hold on, drawnow,
    w(:,k+1) = w(:,k) + 2*alpha*(y(ind(k))-w(:,k)'*x(:,ind(k)))*x(:,ind(k));
end

