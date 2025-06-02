% Load from one folder above the current directory
load('../training/weights_log_tensor_leaky.mat');
load('../training/loss_all_log_leaky.mat');
load('../training/preprocessed_full_data.mat', 'X', 'X_original', 'mean_X', 'std_X');
addpath(fullfile('..', 'training'));

% Add training folder to path
addpath(fullfile('..', 'training'));

% === Setup ===
optimizers = {weights_log.optimizer};
num_optimizers = numel(optimizers);
num_epochs = size(all_loss, 1);
fixed_weights = false;  % set true for fixed update animation

% === Field to track ===
fieldname = 'We_latent';  % e.g., 'We1', 'Wd1', 'We_latent', 'Wd_output'

% === Plot training loss over epochs ===
plot_training_comparison(all_loss, optimizers, num_epochs);

% === Weight evolution visualization ===
if fixed_weights
    W0 = weights_log(1).epoch{1}.(fieldname);  % Access as cell if needed
    total_elements = numel(W0);
    rng(42);  % Reproducibility
    num_updates = 200;
    update_indices = randperm(total_elements, num_updates);

    for i = 1:num_optimizers
        plot_weight_evolution(weights_log, fieldname, i, update_indices);
    end
else
    for i = 1:num_optimizers
        plot_weight_evolution(weights_log, fieldname, i);
    end
end