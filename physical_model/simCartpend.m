clear all, close all, clc

m = 1;
M = 5;
L = 2;
g = -10;
d = 1; 

tspan = 0:.1:20;
y0 = [0; 0; pi; .5];


s = 1; % Pendulum up (b=1)
A = [0 1 0 0;
0 -d/M s*m*g/M 0;
0 0 0 1;
0 -s*d/(M*L) -s*(m+M)*g/(M*L) 0];

B = [0; 1/M; 0; s*1/(M*L)];

eigs = [-1.2; -1.5; -1.1; -1.3];
%rank(ctrb(A,B))
%eig(A)
K = place(A,B,eigs);
[t,y] = ode45(@(t,y)pendcart(y,m,M,L,g,d,-K*(y-[1;0;pi;0])),tspan,y0)

for k=1:length(t)
    drawcartpend(y(k,:),m,M,L);
end
