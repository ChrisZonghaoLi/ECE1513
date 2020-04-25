close all 
clear all

x = readtable('x.csv');
x = table2array(x);

x_norm = readtable('x_norm.csv');
x_norm = table2array(x_norm);

x_cov = cov(x_norm);
[U,S,V] = svd(x_cov);
e = eig(U);
