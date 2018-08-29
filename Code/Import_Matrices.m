%% Import matrices

clear all
close all

% Directories
% dir_2015 = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/2015_year_matrix/';
% dir_2016 = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/2016_year_matrix/';
% dir_2017 = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/2017_year_matrix/';
% dir_All  = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/All_year_matrix/';
% dir_All_additional = '/Users/Alliot/documents/cla project/data/all-data-no-na/projections/';

% Load files
% mat_2015_1d = csvread(strcat(dir_2015,'2015_year_matrix_proj_1d.csv'));
% mat_2015_2d = csvread(strcat(dir_2015,'2015_year_matrix_proj_2d.csv'));
% mat_2015_3d = csvread(strcat(dir_2015,'2015_year_matrix_proj_3d.csv'));
% mat_2015_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/normalized/2015_year_matrix.csv');
% 
% mat_2016_1d = csvread(strcat(dir_2016,'2016_year_matrix_proj_1d.csv'));
% mat_2016_2d = csvread(strcat(dir_2016,'2016_year_matrix_proj_2d.csv'));
% mat_2016_3d = csvread(strcat(dir_2016,'2016_year_matrix_proj_3d.csv'));
% mat_2016_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/normalized/2016_year_matrix.csv');
% 
% mat_2017_1d = csvread(strcat(dir_2017,'2017_year_matrix_proj_1d.csv'));
% mat_2017_2d = csvread(strcat(dir_2017,'2017_year_matrix_proj_2d.csv'));
% mat_2017_3d = csvread(strcat(dir_2017,'2017_year_matrix_proj_3d.csv'));
% mat_2017_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/normalized/2017_year_matrix.csv');
% 
% mat_All_1d = csvread(strcat(dir_All,'All_year_matrix_proj_1d.csv'));
% mat_All_2d = csvread(strcat(dir_All,'All_year_matrix_proj_2d.csv'));
% mat_All_3d = csvread(strcat(dir_All,'All_year_matrix_proj_3d.csv'));
% mat_All_norm = csvread('/Users/Alliot/documents/cla project/data/matrices-no-na/normalized/All_year_matrix.csv');
% 
% mat_All_additional_1d = csvread(strcat(dir_All_additional,'algal_bloom_locations_summaries_norm_proj_1d.csv'));
% mat_All_additional_2d = csvread(strcat(dir_All_additional,'algal_bloom_locations_summaries_norm_proj_2d.csv'));
% mat_All_additional_3d = csvread(strcat(dir_All_additional,'algal_bloom_locations_summaries_norm_proj_3d.csv'));
% mat_All_additional_norm = csvread('/Users/Alliot/documents/cla project/data/all-data-no-na/normalized/algal_bloom_locations_summaries_norm.csv');

dir_summer_2017 = '/Users/Alliot/documents/cla-project/data/matrices-no-na/projections/2017_summer_matrix/';
mat_summer_2017_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized/2017_summer_matrix.csv');
mat_summer_2017_w_ind_3d = csvread(strcat(dir_summer_2017, '2017_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_summer_2017_no_ind_3d = csvread(strcat(dir_summer_2017, '2017_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_summer_2017_randn = '/Users/Alliot/documents/cla-project/data/matrices-no-na/gaussian-randn-projections/2017_summer_matrix/';
mat_summer_2017_randn_w_ind_3d = csvread(strcat(dir_summer_2017_randn, '2017_summer_matrix_proj_randn_w-alg-ind_3d.csv'));
mat_summer_2017_randn_no_ind_3d = csvread(strcat(dir_summer_2017_randn, '2017_summer_matrix_proj_randn_no-alg-ind_3d.csv'));

dir_mendota_summer = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/projections/Mendota_summer_matrix/';
mat_mendota_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized/Mendota_summer_matrix.csv');
mat_mendota_summer_w_ind_3d = csvread(strcat(dir_mendota_summer, 'Mendota_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_mendota_summer_no_ind_3d = csvread(strcat(dir_mendota_summer, 'Mendota_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_mendota_summer_randn = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/gaussian-randn-projections/Mendota_summer_matrix/';
mat_mendota_summer_randn_w_ind_3d = csvread(strcat(dir_mendota_summer_randn, 'Mendota_summer_matrix_proj_randn_w-alg-ind_3d.csv'));
mat_mendota_summer_randn_no_ind_3d = csvread(strcat(dir_mendota_summer_randn, 'Mendota_summer_matrix_proj_randn_no-alg-ind_3d.csv'));

dir_monona_summer = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/projections/Monona_summer_matrix/';
mat_monona_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized/Monona_summer_matrix.csv');
mat_monona_summer_w_ind_3d = csvread(strcat(dir_monona_summer, 'Monona_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_monona_summer_no_ind_3d = csvread(strcat(dir_monona_summer, 'Monona_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_monona_summer_randn = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/gaussian-randn-projections/Monona_summer_matrix/';
mat_monona_summer_randn_w_ind_3d = csvread(strcat(dir_monona_summer_randn, 'Monona_summer_matrix_proj_randn_w-alg-ind_3d.csv'));
mat_monona_summer_randn_no_ind_3d = csvread(strcat(dir_monona_summer_randn, 'Monona_summer_matrix_proj_randn_no-alg-ind_3d.csv'));

dir_all_data_summer = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/All_Data_summer_matrix/';
mat_all_data_summer_norm = csvread('/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/normalized/All_Data_summer_matrix.csv');
mat_all_data_summer_w_ind_3d = csvread(strcat(dir_all_data_summer, 'All_Data_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_all_data_summer_no_ind_3d = csvread(strcat(dir_all_data_summer, 'All_Data_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_mendota_all_data_summer = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/Mendota_All_Data_summer_matrix/';
mat_mendota_all_data_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/all-data-no-na/normalized/Mendota_All_Data_summer_matrix.csv');
mat_mendota_all_data_summer_w_ind_3d = csvread(strcat(dir_mendota_all_data_summer, 'Mendota_All_Data_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_mendota_all_data_summer_no_ind_3d = csvread(strcat(dir_mendota_all_data_summer, 'Mendota_All_Data_summer_matrix_proj_no-alg-ind_3d.csv'));

dir_monona_all_data_summer = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/Monona_All_Data_summer_matrix/';
mat_monona_all_data_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/all-data-no-na/normalized/Monona_All_Data_summer_matrix.csv');
mat_monona_all_data_summer_w_ind_3d = csvread(strcat(dir_monona_all_data_summer, 'Monona_All_Data_summer_matrix_proj_w-alg-ind_3d.csv'));
mat_monona_all_data_summer_no_ind_3d = csvread(strcat(dir_monona_all_data_summer, 'Monona_All_Data_summer_matrix_proj_no-alg-ind_3d.csv'));

%% PCA

dir_pca = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/PCA/';
pca_all_data = csvread(strcat(dir_pca, 'All_Data_summer_matrix_pca.csv'));
pca_mendota = csvread(strcat(dir_pca, 'Mendota_All_Data_summer_matrix_pca.csv'));
pca_monona = csvread(strcat(dir_pca, 'Monona_All_Data_summer_matrix_pca.csv'));

%% Kernel Trick

dir_eigen_no_algae = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/';
mat_all_data_summer_orig_no_ind = csvread(strcat(dir_eigen_no_algae, 'All_data_summer_matrix/All_Data_summer_matrix.csv'));
mat_mendota_orig_no_ind = csvread(strcat(dir_eigen_no_algae, 'Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv'));
mat_monona_orig_no_ind = csvread(strcat(dir_eigen_no_algae, 'Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv'));
mat_all_data_orig_no_ind = csvread(strcat(dir_eigen_no_algae, 'All_Data_matrix/All_Data_matrix.csv'));

%% Save variables to .mat folder

save Matlab_Matrices.mat