function [alpha] = logistica(A, indicadores)
%LOGISTICA Summary of this function goes here
%   Detailed explanation goes here
%   A matriz A tem poucas linhas e muitas colunas
lgch0 = -12;
lgch1 =  12;
m = size(A, 1);
b = zeros(m, 1);
for i = 1:m
    if indicadores(i) == 1
        b(i) = lgch1;
    else
        b(i) = lgch0;
    end
end
alpha = resolve2(A, b);
end