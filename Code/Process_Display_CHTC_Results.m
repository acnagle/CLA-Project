directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/poly/submit/';
% directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/rbf/submit/';
% directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/sig/submit/';

coef_name = 'poly-kernel-coef.csv';

ber_files = dir(strcat(directory, '*.0.csv'));
coefs = csvread(strcat(directory, coef_name));
c = [1 (0.05:0.025:10)];
% c = linspace(0.1, 10, 400);
X = zeros(size(coefs, 1));

for i=1:length(ber_files)
    c_vec = csvread(strcat(directory, ber_files(i).name));
    X(i, :) = c_vec';
end

figure
s = surf(coefs, c, X);
title('BER as a function of C and Coef')
xlabel('coef')
ylabel('C')
s.EdgeColor = 'none';