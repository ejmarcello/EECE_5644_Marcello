function mse = evalMSE(w,x,y)

yhat = w'*x;
e = y-yhat;
mse = mean(e.^2);