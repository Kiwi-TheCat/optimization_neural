% Simplified Autoencoder Training with Modular Optimizer Support
clear; clc; close all
% load delta encoded and Z-score normalized data
load('preprocessed_full_data.mat', 'X_train', 'X_original', 'mean_X', 'std_X');
X_train = X_train(1:384,:);

% Network architecture
input_size = size(X_train, 2);
hidden_size = 200;
latent_size = 200;
num_epochs = 20;
learning_rate = 0.0002;
optimizers = {'sgd', 'adagrad', 'adam'};
%optimizers = {'adam'};
   
% Store comparison results
weights_log = struct(); % logs the weights every 10th epoch over all the training epochs
all_loss = zeros(num_epochs, numel(optimizers));    % logs all the loss curves
final_loss = zeros(1, numel(optimizers));           % logs all the losses over all the epochs
x_train_log = cell(num_epochs, numel(optimizers));  % logs all the data that had been trained on
x_test_log = cell(num_epochs, numel(optimizers));   % logs all data that has been tested on for validation

for o = 1:numel(optimizers)
    optimizer_type = optimizers{o};
    fprintf('Training with %s optimizer...\n', optimizer_type);

    [params, optim, relu, leaky_relu, relu_deriv, leaky_relu_deriv] = setup_network(input_size, hidden_size, latent_size); % leaky_relu with alpha=0.1

    [loss_history, tmp_log, final_loss(o)] = train_autoencoder(X_train, params, optim, relu, relu_deriv, optimizer_type, learning_rate, num_epochs, o); 
    % replace relu with leaky_relu and relu_deriv with leaky_relu_deriv for comparison
    
    weights_log(o).optimizer = tmp_log.optimizer;
    weights_log(o).epoch = tmp_log.epoch;
    all_loss(:, o) = loss_history; % stores the batch-loss (summed losses over all N training samples)/N

end
save('loss_all_log_3.mat', 'all_loss', 'final_loss');
save('weights_log_tensor_3.mat', 'weights_log', 'x_test_log');


% plot the loss curves
figure;
hold on;
colors = lines(numel(optimizers));  % use distinguishable colors

for o = 1:numel(optimizers)
    plot(all_loss(:, o), '-', 'Color', colors(o,:), ...
         'LineWidth', 1.5, 'DisplayName', optimizers{o});
end

xlabel('Epoch');
ylabel('Average Batch Loss');
title('Batch Loss for Different Optimizers over epochs');
legend('Location', 'northeast');
grid on;
hold off;
% Save as PNG
saveas(gcf, 'training_loss_plot.png');




