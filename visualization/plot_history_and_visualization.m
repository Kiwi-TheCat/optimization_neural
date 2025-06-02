% Load from one folder above the current directory
load('../training/weights_log_tensor_leaky.mat');
load('../training/loss_all_log_leaky.mat');
load('../training/preprocessed_full_data.mat', 'X_train', 'X_original', 'mean_X', 'std_X');
addpath(fullfile('..', 'training'));

% Reconstruct optimizer list and epoch count from data
optimizers = {weights_log.optimizer};
num_epochs = size(all_loss, 1);
fixed_weights = 0;

% Define the weight field for visualization
fieldname = 'We_latent'; % Options: 'We1', 'We_latent', 'Wd1', 'Wd_output'


% Call visualization functions
plot_training_comparison(all_loss, optimizers, num_epochs);

% Optionally get update indices for fixed-weight animation
if fixed_weights
    W0 = weights_log(1).epoch(1).(fieldname);
    total_elements = numel(W0);
    rng(42);  % For reproducibility
    num_updates = 200;
    update_indices = randperm(total_elements, num_updates);
    % visualize_weight_animation(weights_log,i, update_indices);

    for i=1:numel(optimizers)
        plot_weight_evolution(weights_log, fieldname,i, update_indices) 
    end
else
    for i=1:numel(optimizers)
        plot_weight_evolution(weights_log, fieldname,i) 
    end
    % visualize_weight_animation(weights_log,i);

end