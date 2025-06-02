function X_hats = reconstruction_over_all_epochs(x_train_sample, best_particle_indices, particle_history, unpack, relu)
% RECONSTRUCTION_OVER_ALL_EPOCHS
%   Reconstructs inputs using best particle from each epoch.
%
% Inputs:
%   - X_data                 : (N x D) normalized input data
%   - best_particle_indices : (E x 1) vector of best particle index per epoch
%   - particle_history       : (E x 1) cell array, each cell contains particle matrix [P x D]
%   - unpack                 : function handle to convert flattened vector to parameter struct
%   - relu                   : activation function (e.g., @relu)
%
% Output:
%   - X_hats                 : (E x 1) cell array of reconstructed data [N x D] per epoch

    num_epochs = length(best_particle_indices);
    X_hats = cell(num_epochs, 1);
    for epoch = 1:num_epochs
        particles = particle_history{epoch};
        best_idx = best_particle_indices(epoch);

        % Skip if index is invalid (e.g., 0 due to early stopping bug)
        if best_idx < 1 || best_idx > size(particles, 1)
            %warning("Skipping epoch %d due to invalid best_idx: %d", epoch, best_idx);
            return;
        end

        best_vector = particles(best_idx, :);
        params = unpack(best_vector);

        % Vectorized forward pass for all samples
        h1 = relu(x_train_sample * params.We1 + params.be1);            
        z  = relu(h1 * params.We_latent + params.be_latent);    
        h2 = relu(z * params.Wd1 + params.bd1);                 
        x_hat = h2 * params.Wd_output + params.bd_output;       

        X_hats{epoch} = x_hat;  % Store reconstruction
    end
end
