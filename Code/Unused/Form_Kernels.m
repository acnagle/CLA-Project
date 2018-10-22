%% Import Matrices

clear all
load Matlab_Matrices.mat

%% Directory Info

dest_path = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/';

if (exist(dest_path, 'dir') == 0)
    mkdir ../Data/all-data-no-na/kernels
end

%% Form the Kernels!

% Get the shapes of each matrix
[all_summmer_m, all_summer_n] = size(mat_all_data_summer_orig_no_ind);
[mendota_m, mendota_n] = size(mat_mendota_orig_no_ind);
[monona_m, monona_n] = size(mat_monona_orig_no_ind);
[all_m, all_n] = size(mat_all_data_orig_no_ind);

d = all_summmer_m;  % Note: all_m == mendota_m == monona_m == 12

% Create kernel for All Data (summer months only)
for k=1:all_summer_n
    x = mat_all_data_summer_orig_no_ind(:, k);
    for i=1:d
        for j=1:i
            x = [x; x(i).*x(j)];
        end
    end
    mat_ker_all_data_summer_no_ind(:, k) = x;
end

% Create kernel for Mendota
for k=1:mendota_n
    x = mat_mendota_orig_no_ind(:, k);
    for i=1:d
        for j=1:i
            x = [x; x(i).*x(j)];
        end
    end
    mat_ker_mendota_no_ind(:, k) = x;
end

% Create kernel for Monona
for k=1:monona_n
    x = mat_monona_orig_no_ind(:, k);
    for i=1:d
        for j=1:i
            x = [x; x(i).*x(j)];
        end
    end
    mat_ker_monona_no_ind(:, k) = x;
end

% Create kernel for All Data (all months)
for k=1:all_n
    x = mat_all_data_orig_no_ind(:, k);
    for i=1:d
        for j=1:i
            x = [x; x(i).*x(j)];
        end
    end
    mat_ker_all_data_no_ind(:, k) = x;
end

%% Write kernels to .csv

csvwrite(strcat(dest_path, 'All_Data_Summer_Kernel_no_ind.csv'), mat_ker_all_data_summer_no_ind)
csvwrite(strcat(dest_path, 'Mendota_Kernel_no_ind.csv'), mat_ker_mendota_no_ind)
csvwrite(strcat(dest_path, 'Monona_Kernel_no_ind.csv'), mat_ker_monona_no_ind)
csvwrite(strcat(dest_path, 'All_Data_Kernel_no_ind.csv'), mat_ker_all_data_no_ind)
