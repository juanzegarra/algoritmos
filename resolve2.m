function [alpha] = resolve2(A,b)
%RESOLVE2 computa alpha: (I + A'*A)*alpha = A'*b sem explicitar a matriz 
%         normal  A'*A
%Referência: artigo [doi ........]
[m, n] = size(A);
In = speye(n);
Im = speye(m);
% In = eye(n);
% Im = eye(m);
bm = [zeros(n,1); b];
M = [In, A'; A,-Im];
x = M\bm;
alpha = [x(1:n)];
end