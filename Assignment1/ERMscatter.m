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

