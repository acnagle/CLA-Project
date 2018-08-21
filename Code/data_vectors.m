function [no_alg, bg_alg, gr_alg] = data_vectors(mat_norm, mat_proj)
% This function accepts the normalized matrix containing all of the data
% points and projection matrix. no_alg, bg_alg, and gr_alg
% matrices are used to visualize the data points corresponding to no algae,
% green algae, and blue-green algae blooms.
    
    % Establish the indices for no algae, green algae and blue green algae
    no_idx = 1;
    bg_idx = 1;
    gr_idx = 1;

    for i=1:length(mat_norm)
        if mat_norm(4, i) == 0
            no_alg(:, no_idx) = mat_proj(:, i);
            no_idx = no_idx + 1;
        elseif mat_norm(4, i) == 0.5
            bg_alg(:, bg_idx) = mat_proj(:, i);
            bg_idx = bg_idx + 1;
        elseif mat_norm(4, i) == 1
            gr_alg(:, gr_idx) = mat_proj(:, i);
            gr_idx = gr_idx + 1;
        end
    end
    
end