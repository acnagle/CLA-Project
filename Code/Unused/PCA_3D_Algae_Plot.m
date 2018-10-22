function PCA_3D_algae_plot(no_alg, bg_alg, gr_alg)
% This function is used to perform the visualization of PCA for the algal
% bloom data. no_alg is a [3, M] matrix for all data points corresponding
% to no algal bloom. bg_alg is a [3, M] matrix for all data points
% corresponding to a blue-green algal bloom. gr_alg is a [3, M] matrix for
% all data points corresponding to a green algal bloom. title is the title
% of the #D plot. legend used to indicate which data points belong to which
% category (no bloom, blue-green bloom, green bloom).

figure
scatter3(no_alg(1, :), no_alg(2, :), no_alg(3, :), 'r')
hold on
scatter3(bg_alg(1, :), bg_alg(2, :), bg_alg(3, :), 'b')
hold on
scatter3(gr_alg(1, :), gr_alg(2, :), gr_alg(3, :), 'g')

end