function [nomes, A] = montaA(seqs)
%montaA6 transforma as sequencias descritas em
%   seqs (obtidas pelo readfasta) em pontos (colunas de A)
%   
m = size(seqs, 1);
n = 20^3;
A = sparse (n, m);
for j = 1:m
    j
    s = seqs(j).Sequence;
    nomes{j} = seqs(j).Header;
    A(:, j) = slidwindow(s);
end
end

