close all
clear all

x1 = [1;1];
x2 = [-1;-1];
x3 = [1;0];
x4 = [0;1];
x = [x1 x2 x3 x4];
j = 4;

Y(:,:,4) = [1;-1;-1;-1];
m = 0:0.1:1.5;
k = - m + 1.5;

figure(1)
p1 = scatter(x(1,:), x(2,:));
set(p1, 'linewidth', 3)
hold on 
p2 = plot(m,k);
set(p2, 'linewidth', 3)
grid on
set(gca,'linewidth',2)

% support point: x1, x3 and x4

X = transpose(x);

% Formulating Q.P.
H = [2 2 -1 -1; 2 2 -1 -1; -1 -1 1 0; -1 -1 0 1];
f = [-1; -1; -1; -1];
lb = zeros(4,1);
Aeq = [1 -1 -1 -1];
beq = 0;

[x,fval,exitflag,output,lambda] = ...
   quadprog(H,f,[],[],Aeq, beq, lb, []);


