% Load from one folder above the current directory
load('../training/weights_log_tensor_3.mat');
load('../training/loss_all_log_3.mat');
load('../training/preprocessed_full_data.mat', 'X_train', 'X_original', 'mean_X', 'std_X');
addpath(fullfile('..', 'training'));

% Reconstruct optimizer list and epoch count from data
optimizers = {weights_log.optimizer};
num_epochs = size(all_loss, 1);
fixed_weights = false;

% Define the weight field for visualization
fieldname = 'We1'; % Options: 'We1', 'We_latent', 'Wd1', 'Wd_output'

% Define validation set (make sure indices are valid)
x_validation = X_train(385:577, :);  % This selects a subset of data for validation

% Initialize activation functions
[~, ~, relu, leaky_relu, ~, ~] = setup_network();

% Compute validation loss from the last saved model of each optimizer
for i = 1:numel(optimizers)
    params = weights_log(i).epoch(end);  % Get last epoch weights for optimizer i
    final_loss(numel(optimizers)+i) = compute_reconstruction_mse(params, x_validation, relu);
end

% Optionally get update indices for fixed-weight animation
if fixed_weights
    W0 = weights_log(1).epoch(1).(fieldname);
    total_elements = numel(W0);
    rng(42);  % For reproducibility
    num_updates = 200;
    update_indices = randperm(total_elements, num_updates);
else
    update_indices = [];  % Pass empty to auto-detect in visualization
end
% Call visualization functions
%plot_training_comparison(all_loss, optimizers, num_epochs);

plot_final_mse(final_loss, [optimizers,'validation']);

visualize_weight_animation(weights_log, x_test_log, update_indices);
%visualize_weight_static(weights_log, x_test_log);
%visualize_weight_evolution(weights_log, x_test_log);
%plot_weight_evolution(weights_log, fieldname, update_indices) 
