%% Import matrices
% Directories
dir_2015 = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/2015_year_matrix/';
dir_2016 = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/2016_year_matrix/';
dir_2017 = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/2017_year_matrix/';
dir_All  = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/All_year_matrix/';
dir_All_additional = '/Users/Alliot/documents/cla project/data/all-data-no-na/projections/';

% Load files
mat_2015_1d = csvread(strcat(dir_2015,'2015_year_matrix_proj_1d.csv'));
mat_2015_2d = csvread(strcat(dir_2015,'2015_year_matrix_proj_2d.csv'));
mat_2015_3d = csvread(strcat(dir_2015,'2015_year_matrix_proj_3d.csv'));
mat_2015_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/eigenvectors/2015_year_matrix/2015_year_matrix.csv');

mat_2016_1d = csvread(strcat(dir_2016,'2016_year_matrix_proj_1d.csv'));
mat_2016_2d = csvread(strcat(dir_2016,'2016_year_matrix_proj_2d.csv'));
mat_2016_3d = csvread(strcat(dir_2016,'2016_year_matrix_proj_3d.csv'));
mat_2016_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/eigenvectors/2016_year_matrix/2016_year_matrix.csv');

mat_2017_1d = csvread(strcat(dir_2017,'2017_year_matrix_proj_1d.csv'));
mat_2017_2d = csvread(strcat(dir_2017,'2017_year_matrix_proj_2d.csv'));
mat_2017_3d = csvread(strcat(dir_2017,'2017_year_matrix_proj_3d.csv'));
mat_2017_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/eigenvectors/2017_year_matrix/2017_year_matrix.csv');

mat_All_1d = csvread(strcat(dir_All,'All_year_matrix_proj_1d.csv'));
mat_All_2d = csvread(strcat(dir_All,'All_year_matrix_proj_2d.csv'));
mat_All_3d = csvread(strcat(dir_All,'All_year_matrix_proj_3d.csv'));
mat_All_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/eigenvectors/All_year_matrix/All_year_matrix.csv');

mat_All_additional_1d = csvread(strcat(dir_All_additional,'algal_bloom_locations_summaries_norm_proj_1d.csv'));
mat_All_additional_2d = csvread(strcat(dir_All_additional,'algal_bloom_locations_summaries_norm_proj_2d.csv'));
mat_All_additional_3d = csvread(strcat(dir_All_additional,'algal_bloom_locations_summaries_norm_proj_3d.csv'));
mat_All_additional_norm = csvread('/Users/Alliot/documents/cla project/data/all-data-no-na/eigenvectors/algal_bloom_locations_summaries_norm.csv');

dir_summer_2017 = '/Users/Alliot/documents/cla-project/data/matrices-no-na/projections/2017_summer_matrix/';
mat_summer_2017_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/2017_summer_matrix/2017_summer_matrix.csv');
% mat_summer_2017_1d = csvread(strcat(dir_summer_2017, 'summer_2017_matrix_proj_1d.csv'));
% mat_summer_2017_2d = csvread(strcat(dir_summer_2017, 'summer_2017_matrix_proj_2d.csv'));
mat_summer_2017_w_ind_3d = csvread(strcat(dir_summer_2017, '2017_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_summer_2017_no_ind_3d = csvread(strcat(dir_summer_2017, '2017_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_summer_2017_randn = '/Users/Alliot/documents/cla-project/data/matrices-no-na/gaussian-randn-projections/2017_summer_matrix/';
mat_summer_2017_randn_w_ind_3d = csvread(strcat(dir_summer_2017_randn, '2017_summer_matrix_proj_randn_w-alg-ind_3d.csv'));
mat_summer_2017_randn_no_ind_3d = csvread(strcat(dir_summer_2017_randn, '2017_summer_matrix_proj_randn_no-alg-ind_3d.csv'));

dir_mendota_summer = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/projections/Mendota_summer_matrix/';
mat_mendota_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/Mendota_summer_matrix/Mendota_summer_matrix.csv');
mat_mendota_summer_w_ind_3d = csvread(strcat(dir_mendota_summer, 'Mendota_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_mendota_summer_no_ind_3d = csvread(strcat(dir_mendota_summer, 'Mendota_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_mendota_summer_randn = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/gaussian-randn-projections/Mendota_summer_matrix/';
mat_mendota_summer_randn_w_ind_3d = csvread(strcat(dir_mendota_summer_randn, 'Mendota_summer_matrix_proj_randn_w-alg-ind_3d.csv'));
mat_mendota_summer_randn_no_ind_3d = csvread(strcat(dir_mendota_summer_randn, 'Mendota_summer_matrix_proj_randn_no-alg-ind_3d.csv'));

dir_monona_summer = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/projections/Monona_summer_matrix/';
mat_monona_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/Monona_summer_matrix/Monona_summer_matrix.csv');
mat_monona_summer_w_ind_3d = csvread(strcat(dir_monona_summer, 'Monona_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_monona_summer_no_ind_3d = csvread(strcat(dir_monona_summer, 'Monona_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_monona_summer_randn = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/gaussian-randn-projections/Monona_summer_matrix/';
mat_monona_summer_randn_w_ind_3d = csvread(strcat(dir_monona_summer_randn, 'Monona_summer_matrix_proj_randn_w-alg-ind_3d.csv'));
mat_monona_summer_randn_no_ind_3d = csvread(strcat(dir_monona_summer_randn, 'Monona_summer_matrix_proj_randn_no-alg-ind_3d.csv'));

%% Save variables to .mat folder

save matlab_matrices.mat