% Load from one folder above the current directory
load('../training/weights_log_tensor.mat', 'weights_log', 'x_test_log');
load('../training/loss_all_log.mat', 'loss_all', 'mse_all');

% Reconstruct optimizer list and epoch count from data
optimizers = {weights_log.optimizer};
num_epochs = size(loss_all, 1);

% Extract the final x_train and x_test for each optimizer (e.g., for the last optimizer)
o = numel(optimizers); % or whichever you want to visualize


% Call visualization functions
%plot_training_comparison(loss_all, optimizers, num_epochs);
%plot_final_mse(mse_all, optimizers);

visualize_weight_animation(weights_log, x_test_log);
visualize_weight_static(weights_log, x_test_log);
%visualize_weight_evolution(weights_log, x_test_log);
