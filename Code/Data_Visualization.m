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

dir_summer_2017 = '/Users/Alliot/documents/cla-project/data/matrices-no-na/projections/summer_2017_matrix/';
mat_summer_2017_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/summer_2017_matrix/summer_2017_matrix.csv');
% mat_summer_2017_1d = csvread(strcat(dir_summer_2017, 'summer_2017_matrix_proj_1d.csv'));
% mat_summer_2017_2d = csvread(strcat(dir_summer_2017, 'summer_2017_matrix_proj_2d.csv'));
mat_summer_2017_3d = csvread(strcat(dir_summer_2017, 'summer_2017_matrix_proj_3d.csv'));

dir_summer_2017_randn = '/Users/Alliot/documents/cla-project/data/matrices-no-na/gaussian-randn-projections/summer_2017_matrix/';
mat_summer_2017_3d_randn = csvread(strcat(dir_summer_2017_randn, 'summer_2017_matrix_proj_3d.csv'));

dir_mendota_summer = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/projections/Mendota_summer_matrix/';
mat_mendota_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/Mendota_summer_matrix/Mendota_summer_matrix.csv');
mat_mendota_summer_3d = csvread(strcat(dir_mendota_summer, 'Mendota_summer_matrix_proj_3d.csv'));

dir_mendota_summer_randn = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/gaussian-randn-projections/Mendota_summer_matrix/';
mat_mendota_summer_3d_randn = csvread(strcat(dir_mendota_summer_randn, 'Mendota_summer_matrix_proj_3d.csv'));

dir_monona_summer = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/projections/Monona_summer_matrix/';
mat_monona_summer_3d = csvread(strcat(dir_monona_summer, 'Monona_summer_matrix_proj_3d.csv'));

dir_monona_summer_randn = '/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/gaussian-randn-projections/Monona_summer_matrix/';
mat_monona_summer_norm = csvread('/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/Monona_summer_matrix/Monona_summer_matrix.csv');
mat_monona_summer_3d_randn = csvread(strcat(dir_monona_summer_randn, 'Monona_summer_matrix_proj_3d.csv'));

%% Compute vectors to plot

for i=1:length(mat_summer_2017_norm)
    if mat_summer_2017_norm(2, i) == 0
%         no_alg_1d_summer_2017(:, i) = mat_summer_2017_1d(:, i);
%         no_alg_2d_summer_2017(:, i) = mat_summer_2017_2d(:, i);
        no_alg_3d_summer_2017(:, i) = mat_summer_2017_3d(:, i);
    elseif mat_summer_2017_norm(2, i) == 0.5
%         bg_alg_1d_summer_2017(:, i) = mat_summer_2017_1d(:, i);
%         bg_alg_2d_summer_2017(:, i) = mat_summer_2017_2d(:, i);
        bg_alg_3d_summer_2017(:, i) = mat_summer_2017_3d(:, i);
    elseif mat_summer_2017_norm(2, i) == 1
%         gr_alg_1d_summer_2017(:, i) = mat_summer_2017_1d(:, i);
%         gr_alg_2d_summer_2017(:, i) = mat_summer_2017_2d(:, i);
        gr_alg_3d_summer_2017(:, i) = mat_summer_2017_3d(:, i);
    end
end

for i=1:length(mat_mendota_summer_norm)
    if mat_mendota_summer_norm(2, i) == 0
        no_alg_3d_mendota_summer(:, i) = mat_mendota_summer_3d(:, i);
        no_alg_3d_mendota_summer_randn(:, i) = mat_mendota_summer_3d_randn(:, i);
    elseif mat_mendota_summer_norm(2, i) == 0.5
        bg_alg_3d_mendota_summer(:, i) = mat_mendota_summer_3d(:, i);
        bg_alg_3d_mendota_summer_randn(:, i) = mat_mendota_summer_3d_randn(:, i);
    elseif mat_mendota_summer_norm(2, i) == 1
        gr_alg_3d_mendota_summer(:, i) = mat_mendota_summer_3d(:, i);
        gr_alg_3d_mendota_summer_randn(:, i) = mat_mendota_summer_3d_randn(:, i);
    end
end

for i=1:length(mat_monona_summer_norm)
    if mat_monona_summer_norm(2, i) == 0
        no_alg_3d_monona_summer(:, i) = mat_monona_summer_3d(:, i);
        no_alg_3d_monona_summer_randn(:, i) = mat_monona_summer_3d_randn(:, i);
    elseif mat_monona_summer_norm(2, i) == 0.5
        bg_alg_3d_monona_summer(:, i) = mat_monona_summer_3d(:, i);
        bg_alg_3d_monona_summer_randn(:, i) = mat_monona_summer_3d_randn(:, i);
    elseif mat_monona_summer_norm(2, i) == 1
        gr_alg_3d_monona_summer(:, i) = mat_monona_summer_3d(:, i);
        gr_alg_3d_monona_summer_randn(:, i) = mat_monona_summer_3d_randn(:, i);
    end
end

% for i=1:length(mat_All_additional_norm)
%     if mat_All_additional_norm(2, i) == 0
%         no_alg_1d(:, i) = mat_All_additional_1d(:, i);
%         no_alg_2d(:, i) = mat_All_additional_2d(:, i);
%         no_alg_3d(:, i) = mat_All_additional_3d(:, i);
%     elseif mat_All_additional_norm(2, i) == 0.5
%         bg_alg_1d(:, i) = mat_All_additional_1d(:, i);
%         bg_alg_2d(:, i) = mat_All_additional_2d(:, i);
%         bg_alg_3d(:, i) = mat_All_additional_3d(:, i);
%     elseif mat_All_additional_norm(2, i) == 1
%         gr_alg_1d(:, i) = mat_All_additional_1d(:, i);
%         gr_alg_2d(:, i) = mat_All_additional_2d(:, i);
%         gr_alg_3d(:, i) = mat_All_additional_3d(:, i);
%     end
% end
% 
% for i=1:length(mat_2015_norm)
%     if mat_2015_norm(2, i) == 0
%         no_alg_1d_2015(:, i) = mat_2015_1d(:, i);
%         no_alg_2d_2015(:, i) = mat_2015_2d(:, i);
%         no_alg_3d_2015(:, i) = mat_2015_3d(:, i);
%     elseif mat_2015_norm(2, i) == 0.5
%         bg_alg_1d_2015(:, i) = mat_2015_1d(:, i);
%         bg_alg_2d_2015(:, i) = mat_2015_2d(:, i);
%         bg_alg_3d_2015(:, i) = mat_2015_3d(:, i);
%     elseif mat_2015_norm(2, i) == 1
%         gr_alg_1d_2015(:, i) = mat_2015_1d(:, i);
%         gr_alg_2d_2015(:, i) = mat_2015_2d(:, i);
%         gr_alg_3d_2015(:, i) = mat_2015_3d(:, i);
%     end
% end
% 
% for i=1:length(mat_2016_norm)
%     if mat_2016_norm(2, i) == 0
%         no_alg_1d_2016(:, i) = mat_2016_1d(:, i);
%         no_alg_2d_2016(:, i) = mat_2016_2d(:, i);
%         no_alg_3d_2016(:, i) = mat_2016_3d(:, i);
%     elseif mat_2016_norm(2, i) == 0.5
%         bg_alg_1d_2016(:, i) = mat_2016_1d(:, i);
%         bg_alg_2d_2016(:, i) = mat_2016_2d(:, i);
%         bg_alg_3d_2016(:, i) = mat_2016_3d(:, i);
%     elseif mat_2016_norm(2, i) == 1
%         gr_alg_1d_2016(:, i) = mat_2016_1d(:, i);
%         gr_alg_2d_2016(:, i) = mat_2016_2d(:, i);
%         gr_alg_3d_2016(:, i) = mat_2016_3d(:, i);
%     end
% end
% 
% for i=1:length(mat_2017_norm)
%     if mat_2017_norm(2, i) == 0
%         no_alg_1d_2017(:, i) = mat_2017_1d(:, i);
%         no_alg_2d_2017(:, i) = mat_2017_2d(:, i);
%         no_alg_3d_2017(:, i) = mat_2017_3d(:, i);
%     elseif mat_2017_norm(2, i) == 0.5
%         bg_alg_1d_2017(:, i) = mat_2017_1d(:, i);
%         bg_alg_2d_2017(:, i) = mat_2017_2d(:, i);
%         bg_alg_3d_2017(:, i) = mat_2017_3d(:, i);
%     elseif mat_2017_norm(2, i) == 1
%         gr_alg_1d_2017(:, i) = mat_2017_1d(:, i);
%         gr_alg_2d_2017(:, i) = mat_2017_2d(:, i);
%         gr_alg_3d_2017(:, i) = mat_2017_3d(:, i);
%     end
% end
% 
% for i=1:length(mat_All_norm)
%     if mat_All_norm(2, i) == 0
%         no_alg_1d_All(:, i) = mat_All_1d(:, i);
%         no_alg_2d_All(:, i) = mat_All_2d(:, i);
%         no_alg_3d_All(:, i) = mat_All_3d(:, i);
%     elseif mat_All_norm(2, i) == 0.5
%         bg_alg_1d_All(:, i) = mat_All_1d(:, i);
%         bg_alg_2d_All(:, i) = mat_All_2d(:, i);
%         bg_alg_3d_All(:, i) = mat_All_3d(:, i);
%     elseif mat_All_norm(2, i) == 1
%         gr_alg_1d_All(:, i) = mat_All_1d(:, i);
%         gr_alg_2d_All(:, i) = mat_All_2d(:, i);
%         gr_alg_3d_All(:, i) = mat_All_3d(:, i);
%     end
% end

%% Plot Summer 2017 Data Points

figure
scatter3(no_alg_3d_summer_2017(1, :), no_alg_3d_summer_2017(2, :), no_alg_3d_summer_2017(3, :), 'r')
hold on
scatter3(bg_alg_3d_summer_2017(1, :), bg_alg_3d_summer_2017(2, :), bg_alg_3d_summer_2017(3, :), 'b')
hold on
scatter3(gr_alg_3d_summer_2017(1, :), gr_alg_3d_summer_2017(2, :), gr_alg_3d_summer_2017(3, :), 'g')
title('Summer 2017 3D')
legend('no algae', 'blue-green algae', 'green algae')

% figure
% scatter(no_alg_2d_summer_2017(1, :), no_alg_2d_summer_2017(2, :), 'r')
% hold on
% scatter(bg_alg_2d_summer_2017(1, :), bg_alg_2d_summer_2017(2, :), 'b')
% hold on
% scatter(gr_alg_2d_summer_2017(1, :), gr_alg_2d_summer_2017(2, :), 'g')
% title('Summer 2017 2D')
% legend('no algae', 'blue-green algae', 'green algae')
% 
% figure
% for i=1:length(mat_summer_2017_norm)
%     if mat_summer_2017_norm(2, i) == 0
%         plot(no_alg_1d_summer_2017(i), 'or')
%     elseif mat_summer_2017_norm(2, i) == 0.5
%         plot(bg_alg_1d_summer_2017(i), 'ob')
%     elseif mat_summer_2017_norm(2, i) == 1
%         plot(gr_alg_1d_summer_2017(i), 'og')
%     end
%     hold on
% end
% title('Summer 2017 1D')

%% Plot Mendota Summer Data Points

figure
scatter3(no_alg_3d_mendota_summer(1, :), no_alg_3d_mendota_summer(2, :), no_alg_3d_mendota_summer(3, :), 'r')
hold on
scatter3(bg_alg_3d_mendota_summer(1, :), bg_alg_3d_mendota_summer(2, :), bg_alg_3d_mendota_summer(3, :), 'b')
hold on
scatter3(gr_alg_3d_mendota_summer(1, :), gr_alg_3d_mendota_summer(2, :), gr_alg_3d_mendota_summer(3, :), 'g')
title('Mendota Summer (June-August) 3D')
legend('no algae', 'blue-green algae', 'green algae')

%% Plot Monona Summer Data Points

figure
scatter3(no_alg_3d_monona_summer(1, :), no_alg_3d_monona_summer(2, :), no_alg_3d_monona_summer(3, :), 'r')
hold on
scatter3(bg_alg_3d_monona_summer(1, :), bg_alg_3d_monona_summer(2, :), bg_alg_3d_monona_summer(3, :), 'b')
hold on
scatter3(gr_alg_3d_monona_summer(1, :), gr_alg_3d_monona_summer(2, :), gr_alg_3d_monona_summer(3, :), 'g')
title('Monona Summer (June-August) 3D')
legend('no algae', 'blue-green algae', 'green algae')

%% Plot All Data points (with additional measurements)
% 
% figure
% scatter3(no_alg_3d(1, :), no_alg_3d(2, :), no_alg_3d(3, :), 'r')
% hold on
% scatter3(bg_alg_3d(1, :), bg_alg_3d(2, :), bg_alg_3d(3, :), 'b')
% hold on
% scatter3(gr_alg_3d(1, :), gr_alg_3d(2, :), gr_alg_3d(3, :), 'g')
% title('All Data 3D (additional measurements)')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% scatter(no_alg_2d(1, :), no_alg_2d(2, :), 'r')
% hold on
% scatter(bg_alg_2d(1, :), bg_alg_2d(2, :), 'b')
% hold on
% scatter(gr_alg_2d(1, :), gr_alg_2d(2, :), 'g')
% title('All Data 2D (additional measurements)')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% for i=1:length(mat_2017_norm)
%     if mat_All_additional_norm(2, i) == 0
%         plot(no_alg_1d(i), 'or')
%     elseif mat_All_additional_norm(2, i) == 0.5
%         plot(bg_alg_1d(i), 'ob')
%     elseif mat_All_additional_norm(2, i) == 1
%         plot(gr_alg_1d(i), 'og')
%     end
%     hold on
% end
% title('All Data 1D (additional measurements)')
% 
% 
% %% Plot 2015 Data Points
% 
% figure
% scatter3(no_alg_3d_2015(1, :), no_alg_3d_2015(2, :), no_alg_3d_2015(3, :), 'r')
% hold on
% scatter3(bg_alg_3d_2015(1, :), bg_alg_3d_2015(2, :), bg_alg_3d_2015(3, :), 'b')
% hold on
% scatter3(gr_alg_3d_2015(1, :), gr_alg_3d_2015(2, :), gr_alg_3d_2015(3, :), 'g')
% title('2015 3D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% scatter(no_alg_2d_2015(1, :), no_alg_2d_2015(2, :), 'r')
% hold on
% scatter(bg_alg_2d_2015(1, :), bg_alg_2d_2015(2, :), 'b')
% hold on
% scatter(gr_alg_2d_2015(1, :), gr_alg_2d_2015(2, :), 'g')
% title('2015 2D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% for i=1:length(mat_2015_norm)
%     if mat_2015_norm(2, i) == 0
%         plot(no_alg_1d_2015(i), 'or')
%     elseif mat_2015_norm(2, i) == 0.5
%         plot(bg_alg_1d_2015(i), 'ob')
%     elseif mat_2015_norm(2, i) == 1
%         plot(gr_alg_1d_2015(i), 'og')
%     end
%     hold on
% end
% title('2015 1D')
% 
% %% Plot 2016 Data Points
% 
% figure
% scatter3(no_alg_3d_2016(1, :), no_alg_3d_2016(2, :), no_alg_3d_2016(3, :), 'r')
% hold on
% scatter3(bg_alg_3d_2016(1, :), bg_alg_3d_2016(2, :), bg_alg_3d_2016(3, :), 'b')
% hold on
% scatter3(gr_alg_3d_2016(1, :), gr_alg_3d_2016(2, :), gr_alg_3d_2016(3, :), 'g')
% title('2016 3D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% scatter(no_alg_2d_2016(1, :), no_alg_2d_2016(2, :), 'r')
% hold on
% scatter(bg_alg_2d_2016(1, :), bg_alg_2d_2016(2, :), 'b')
% hold on
% scatter(gr_alg_2d_2016(1, :), gr_alg_2d_2016(2, :), 'g')
% title('2016 2D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% for i=1:length(mat_2016_norm)
%     if mat_2016_norm(2, i) == 0
%         plot(no_alg_1d_2016(i), 'or')
%     elseif mat_2016_norm(2, i) == 0.5
%         plot(bg_alg_1d_2016(i), 'ob')
%     elseif mat_2016_norm(2, i) == 1
%         plot(gr_alg_1d_2016(i), 'og')
%     end
%     hold on
% end
% title('2016 1D')
% 
% %% Plot 2017 Data Points
% 
% figure
% scatter3(no_alg_3d_2017(1, :), no_alg_3d_2017(2, :), no_alg_3d_2017(3, :), 'r')
% hold on
% scatter3(bg_alg_3d_2017(1, :), bg_alg_3d_2017(2, :), bg_alg_3d_2017(3, :), 'b')
% hold on
% scatter3(gr_alg_3d_2017(1, :), gr_alg_3d_2017(2, :), gr_alg_3d_2017(3, :), 'g')
% title('2017 3D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% scatter(no_alg_2d_2017(1, :), no_alg_2d_2017(2, :), 'r')
% hold on
% scatter(bg_alg_2d_2017(1, :), bg_alg_2d_2017(2, :), 'b')
% hold on
% scatter(gr_alg_2d_2017(1, :), gr_alg_2d_2017(2, :), 'g')
% title('2017 2D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% for i=1:length(mat_2017_norm)
%     if mat_2017_norm(2, i) == 0
%         plot(no_alg_1d_2017(i), 'or')
%     elseif mat_2017_norm(2, i) == 0.5
%         plot(bg_alg_1d_2017(i), 'ob')
%     elseif mat_2017_norm(2, i) == 1
%         plot(gr_alg_1d_2017(i), 'og')
%     end
%     hold on
% end
% title('2017 1D')
% 
%% Plot All Data Points
% 
% figure
% scatter3(no_alg_3d_All(1, :), no_alg_3d_All(2, :), no_alg_3d_All(3, :), 'r')
% hold on
% scatter3(bg_alg_3d_All(1, :), bg_alg_3d_All(2, :), bg_alg_3d_All(3, :), 'b')
% hold on
% scatter3(gr_alg_3d_All(1, :), gr_alg_3d_All(2, :), gr_alg_3d_All(3, :), 'g')
% title('All 3D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% scatter(no_alg_2d_All(1, :), no_alg_2d_All(2, :), 'r')
% hold on
% scatter(bg_alg_2d_All(1, :), bg_alg_2d_All(2, :), 'b')
% hold on
% scatter(gr_alg_2d_All(1, :), gr_alg_2d_All(2, :), 'g')
% title('All 2D')
% legend('no algae', 'blue-green algae','green algae')
% 
% figure
% for i=1:length(mat_All_norm)
%     if mat_All_norm(2, i) == 0
%         plot(no_alg_1d_All(i), 'or')
%     elseif mat_All_norm(2, i) == 0.5
%         plot(bg_alg_1d_All(i), 'ob')
%     elseif mat_All_norm(2, i) == 1
%         plot(gr_alg_1d_All(i), 'og')
%     end
%     hold on
% end
% title('All 1D')