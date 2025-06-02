% Simplified Autoencoder Training with Modular Optimizer Support
% Minimal Autoencoder Architecture: 4 Layers
% Encoder (2 layers):
% Input layer — raw input (e.g., 121’104 for a 384×384 image)
% Hidden layer — compressed representation (e.g., 200 or 300 units)
% 
% Decoder (2 layers):
% Hidden layer — expands back from latent space
% Output layer — reconstructs input (same size as original input)

% x (input) → [ENCODER] → h (latent) → [DECODER] → x̂ (reconstructed)

% Simplified Autoencoder Training with Modular Optimizer Support
clear; clc; close all
addpath(fullfile('..', 'training')); % like this the functions from training become usable

% load delta encoded and Z-score normalized data
load('../training/preprocessed_full_data.mat', 'X_train', 'X_original', 'mean_X', 'std_X');
X_train = X_train(1:384,:);

%% Network architecture
input_size = size(X_train, 2);
hidden_size = 200;
latent_size = 200;
num_epochs = 20;
optimizers = {'PSO1', 'PSO2', 'PSO3', 'PSO4'};
pso_params = [];
% === PSO parameters ===
pso_params.w = 0.5; % momentum
pso_params.c1 = 1.5; % cognitive
pso_params.c2 = 0.3; % social
num_particles = 30;

%% Store comparison results
weights_log = struct();
all_loss = zeros(num_epochs, numel(optimizers));
mse_all = zeros(1, numel(optimizers));
x_train_log = cell(num_epochs, numel(optimizers));
x_test_log = cell(num_epochs, numel(optimizers));
p_best_loss_total = 0;

for o = 1:numel(optimizers)
    optimizer_type = optimizers{o};
    fprintf('Training with %s optimizer...\n', optimizer_type);

    [params, optim, relu, relu_deriv] = setup_network(input_size, hidden_size, latent_size);

    [loss_history, tmp_log, mse_all(o), p_best_loss] = train_autoencoder_pso(X_train, params, relu, optimizers{o}, num_epochs, num_particles, pso_params);
    string = sprintf("personal bests (first 5): %d, %d, %d, %d, %d", p_best_loss(1), p_best_loss(2), p_best_loss(3), p_best_loss(4), p_best_loss(5));
    disp(string)
    weights_log(o).optimizer = tmp_log.optimizer;
    weights_log(o).epoch = tmp_log.epoch;
    all_loss(:, o) = loss_history;
end


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

