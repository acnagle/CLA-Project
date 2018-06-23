function [no_alg, bg_alg, gr_alg] = data_vectors(mat_norm, mat_proj)
% This function accepts the normalized matrix containing all of the data
% points and projection matrix. no_alg, bg_alg, and gr_alg
% matrices are used to visualize the data points corresponding to no algae,
% green algae, and blue-green algae blooms.

    for i=1:length(mat_norm)
        if mat_norm(2, i) == 0
            no_alg(:, i) = mat_proj(:, i);
        elseif mat_norm(2, i) == 0.5
            bg_alg(:, i) = mat_proj(:, i);
        elseif mat_norm(2, i) == 1
            gr_alg(:, i) = mat_proj(:, i);
        end
    end
    
end