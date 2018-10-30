directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/poly/';

ber_files = dir(strcat(directory, '*.0.csv'));
coefs = csvread(strcat(directory, 'poly-kernel-coef.csv'));

X = zeros(size(coefs, 1));

for i=1:length(ber_files)
    c_vec = csvread(strcat(directory, ber_files(i).name));
    X(i, :) = c_vec;
end

figure
surf(coefs, coefs, X)
title('BER as a function of C and Coef')
xlabel('coef')
ylabel('C')
