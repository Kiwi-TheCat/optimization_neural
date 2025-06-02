% Simplified Autoencoder Training with Modular Optimizer Support
clear; clc; close all
% load delta encoded and Z-score normalized data
load('preprocessed_full_data.mat', 'X', 'X_original', 'mean_X', 'std_X');
num_samples = size(X,1);
num_samples_train = ceil(num_samples*0.8);
X_train = X(1:num_samples_train,:);

% Network architecture
input_size = size(X_train, 2);
hidden_size = 200;
latent_size = 200;
num_epochs = 20;
learning_rate = 0.002; % should be low for stochastic gradient descend (~0.0002)
regularization_lambda = 0.0001; % determines L2 regularization lambda
alpha_leaky = 0.1; % determines leakiness for leaky_relu
batch_descend = 1; 
batch_size = 300; 
% true makes the optim update once per batch (more generalized), 
% false once per sample (sgd is always sample wise)
optimizers = {'sgd', 'adagrad', 'adam'};
   
% Store comparison results
weights_log = struct(); % logs the weights every 10th epoch over all the training epochs
num_batches = ceil(num_samples_train/batch_size);
num_losses_tracked = num_epochs*num_batches ;
all_loss = zeros(num_losses_tracked, numel(optimizers));    % logs all the loss curves
all_loss_per_epoch = zeros(num_epochs, numel(optimizers));    % logs all the loss curves
final_loss = zeros(1, numel(optimizers));           % logs all the losses over all the epochs
x_train_log = cell(num_epochs, numel(optimizers));  % logs all the data that had been trained on

for o = 1:numel(optimizers)
    optimizer_type = optimizers{o};
    fprintf('Training with %s optimizer...\n', optimizer_type);

    [params, optim, relu, leaky_relu, relu_deriv, leaky_relu_deriv] = setup_network(input_size, hidden_size, latent_size, alpha_leaky); 

    [loss_history_after_each_update, loss_history_per_epoch, tmp_log, final_loss(o)] = train_autoencoder(X_train, params, optim, leaky_relu, leaky_relu_deriv,...
        optimizer_type, learning_rate, num_epochs, regularization_lambda, batch_descend, batch_size); 
    % replace relu with leaky_relu and relu_deriv with leaky_relu_deriv for comparison
    
    weights_log(o).optimizer = tmp_log.optimizer;
    weights_log(o).epoch = tmp_log.epoch;
    all_loss(:, o) = loss_history_after_each_update; % stores the batch-loss (summed losses over all N training samples)/N
    all_loss_per_epoch(:, o) = loss_history_per_epoch; % stores the batch-loss (summed losses over all N training samples)/N
end
% save('loss_all_log_leaky.mat', 'all_loss', 'final_loss');
% save('weights_log_tensor_leaky.mat', 'weights_log');

save('loss_all_log_leaky_fixed.mat', 'all_loss', 'final_loss');
save('weights_log_tensor_leaky_fixed.mat', 'weights_log');

% plot the loss curves
figure;
hold on;
colors = lines(numel(optimizers));  % use distinguishable colors

for o = 1:numel(optimizers)
    plot(all_loss(:, o), '-', 'Color', colors(o,:), ...
         'LineWidth', 1.5, 'DisplayName', optimizers{o});
end

xlabel('Epoch*Batch');
ylabel('Average Batch Loss');
title('Batch Loss for Different Optimizers over epochs');
legend('Location', 'northeast');
grid on;
hold off;
% Save as PNG
%saveas(gcf, 'training_loss_plot.png');
saveas(gcf, 'training_loss_plot_fixed.png');



% plot the loss curves
figure;
hold on;
colors = lines(numel(optimizers));  % use distinguishable colors

for o = 1:numel(optimizers)
    plot(all_loss_per_epoch(:, o), '-', 'Color', colors(o,:), ...
         'LineWidth', 1.5, 'DisplayName', optimizers{o});
end

xlabel('Epoch');
ylabel('Average Batch Loss');
title('Batch Loss for Different Optimizers over epochs');
legend('Location', 'northeast');
grid on;
hold off;
% Save as PNG
%saveas(gcf, 'training_loss_plot_Epoch.png');
saveas(gcf, 'training_loss_plot_Epoch_fixed.png');



