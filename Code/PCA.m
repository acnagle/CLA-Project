%% Import Matrices

load matlab_matrices.mat

%% Compute vectors to plot

% All Data summer (2014-2017)
[no_alg_3d_all_data_summer_no_ind, bg_alg_3d_all_data_summer_no_ind, ...
    gr_alg_3d_all_data_summer_no_ind] = Data_Vectors(mat_all_data_summer_norm, ...
    pca_all_data);

% All Data mendota summer (2014-2017)
[no_alg_3d_mendota_all_data_summer_no_ind, bg_alg_3d_mendota_all_data_summer_no_ind, ...
    gr_alg_3d_mendota_all_data_summer_no_ind] = Data_Vectors(mat_mendota_all_data_summer_norm, ...
    pca_mendota);

% All Data monona summer (2014-2017)
[no_alg_3d_monona_all_data_summer_no_ind, bg_alg_3d_monona_all_data_summer_no_ind, ...
    gr_alg_3d_monona_all_data_summer_no_ind] = Data_Vectors(mat_monona_all_data_summer_norm, ...
    pca_monona);

%% Plot All Data Summer Points

PCA_3D_Algae_Plot(no_alg_3d_all_data_summer_no_ind, bg_alg_3d_all_data_summer_no_ind, ...
    gr_alg_3d_all_data_summer_no_ind)
title('All Data Summer (June-August) PCA, No Indicator')
legend('no algae', 'blue-green algae', 'green algae')

%% Plot All Data Mendota Points

PCA_3D_Algae_Plot(no_alg_3d_mendota_all_data_summer_no_ind, bg_alg_3d_mendota_all_data_summer_no_ind, ...
    gr_alg_3d_mendota_all_data_summer_no_ind)
title('Mendota All Data Summer (June-August) PCA, No Indicator')
legend('no algae', 'blue-green algae', 'green algae')

%% Plot All Data Monona Points

PCA_3D_Algae_Plot(no_alg_3d_monona_all_data_summer_no_ind, bg_alg_3d_monona_all_data_summer_no_ind, ...
    gr_alg_3d_monona_all_data_summer_no_ind)
title('Monona All Data Summer (June-August) PCA, No Indicator')
legend('no algae', 'blue-green algae', 'green algae')
