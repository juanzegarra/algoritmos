function s = inversa(endereco)
% computa a sequencia correspondente ao endereco
alfabeto = 'ADQIMSYRCGLFTVNEHKPWX';
p = zeros (1, 3);
k = endereco;
p(1) = floor(k/400);
if (k - p(1)*400) > 1.00E-10 
  p(1) = p(1) + 1;
end
k = k - (p(1) - 1)*400;
p(2) = floor(k/20);
if (k - p(2)*20) > 1.00E-10
  p(2) =  p(2) + 1;
end
k = k - (p(2) - 1)*20;
p(3) = k;
if p(3) == 0
  p(3) = 1;
end
s = '';

for i = 1:3
    s = [s, alfabeto(p(i))];
end

end
