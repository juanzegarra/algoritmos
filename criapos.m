function [endereco] = criapos(janela)
alfabeto = 'ADQIMSYRCGLFTVNEHKPWX';
i = find(alfabeto==janela(1));
j = find(alfabeto==janela(2));
k = find(alfabeto==janela(3));

if (i <= 20)&(j <= 20)&(k<= 20)
    endereco = 400*(i-1)+20*(j-1)+k;
else
    endereco = -1;
    disp ('resíduo inválido...........')
end
end

