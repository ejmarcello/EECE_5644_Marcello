%% Regression Demo Script (usage of Neural Networks in Matlab)
% RegressionDemo.m
% Regression is just estimating some relationship between dependent
% variables (function output, vertical axis) and independent variables (horiz axis)
%
% Ethan Marcello
% From: https://www.mathworks.com/help/thingspeak/create-and-train-a-feedforward-neural-network.html
%
%% Regression using a feedforward neural network
clear all; close all;
%ThingSpeak™ channel 12397 contains data from the MathWorks® weather station,
% located in Natick, Massachusetts. The data is collected once every minute.
% Fields 2, 3, 4, and 6 contain wind speed (mph), relative humidity,
% temperature (F), and atmospheric pressure (inHg) data, respectively.
%  Read the data from channel 12397 using the thingSpeakRead function.
data = thingSpeakRead(12397,'Fields',[2 3 4 6],'DateRange',[datetime('January 7, 2018'),datetime('January 9, 2018')],...
    'outputFormat','table');

% For neural network
inputs = [data.Humidity'; data.TemperatureF'; data.PressureHg'; data.WindSpeedmph'];
tempC = (5/9)*(data.TemperatureF-32);
b = 17.62;
c = 243.5;
gamma = log(data.Humidity/100) + b*tempC ./ (c+tempC);
dewPointC = c*gamma ./ (b-gamma);
dewPointF = (dewPointC*1.8) + 32;
targets = dewPointF';

% one hidden layer with 10 neurons
net = feedforwardnet(10);
% train the network to calculate dewPointF
[net,tr] = train(net,inputs,targets);

% Get the output of the network from a given input
output = net(inputs(:,5))

