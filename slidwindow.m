function [frequencia] = slidwindow(s)
frequencia = sparse(8000,1);
[m,n]=size(s);
for i=1:(n-(3-1))
    %janela = [s(i),s(i+1), s(i+2)];
    janela = strcat(s(i),s(i+1), s(i+2));
    p = criapos(janela);
    % desconsidera janelas que contem 'X'
    if p ~= -1
        frequencia(p) = frequencia(p)+1;
    end
end

