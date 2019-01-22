%% RBF Kernel

% directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/rbf/testing,c=1.4.250,gamma=1.4.251/';
directory = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/rbf/testing,c=1.1.250,gamma=1.1.251/';

c_name = 'rbf-kernel-c.csv';

spacing = 1;
stopping_point = 250;

ber_files = dir(strcat(directory, '*.0.csv'));
c = csvread(strcat(directory, c_name));
gamma = [1 (spacing:spacing:1000)];
X = zeros(size(c, 1));

% Each row has a fixed gamma value and a changing C value
for i=1:length(ber_files)
    ber_vec = csvread(strcat(directory, ber_files(i).name));
    X(i, :) = ber_vec';
end

figure
s = surf(X(1:249, :));
title('BER as a function of gamma and C')
xlabel('gamma')
ylabel('C')
s.EdgeColor = 'none';

%% Polynomial Kernel

% directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/poly/submit/';
directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/rbf/previous5/';
% directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/sig/submit/';

coef_name = 'rbf-kernel-coef.csv';

ber_files = dir(strcat(directory, '*.0.csv'));
coefs = csvread(strcat(directory, coef_name));
c = [1 (25:25:10000)];
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

%% Sigmoid Kernel

directory = '/Users/Alliot/documents/cla-project/data/all-data-no-na/kernels/sig/submit/';

coef_name = 'sig-kernel-coef.csv';

ber_files = dir(strcat(directory, '*.0.csv'));
coefs = csvread(strcat(directory, coef_name));
c = [1 (25:25:10000)];
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