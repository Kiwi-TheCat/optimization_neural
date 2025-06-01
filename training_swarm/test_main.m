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


clear; clc; close all
load('data_1s.mat')

%% Parameters
n = 10;
chunk_size = 384;
n_channel = 384;
raw = data(1:n*chunk_size, 1:n_channel);
delta = diff(raw);

%% Select training data
num_train_samples = 1; %384
X_train_original = double(delta(1:num_train_samples, :));

%% Z-score normalization
mean_X = mean(X_train_original);
std_X = std(X_train_original);
std_X(std_X == 0) = 1e-6;
X_train = (X_train_original - mean_X) ./ std_X;

%% Network architecture
input_size = size(X_train, 2);
hidden_size = 200;
latent_size = 200;
num_epochs = 200;
learning_rate = 0.0002;
optimizers = {'PSO', 'PSO', 'PSO', 'PSO', 'PSO'};

%% Store comparison results
weights_log = struct();
loss_all = zeros(num_epochs, numel(optimizers));
mse_all = zeros(1, numel(optimizers));
x_train_log = cell(num_epochs, numel(optimizers));
x_test_log = cell(num_epochs, numel(optimizers));
p_best_loss_total = 0;

for o = 1:numel(optimizers)
    optimizer_type = optimizers{o};
    fprintf('Training with %s optimizer...\n', optimizer_type);

    [params, optim, relu, relu_deriv] = setup_network(input_size, hidden_size, latent_size);

    [loss_history, tmp_log, mse_all(o), x_test_log(:,o), p_best_loss] = train_autoencoder_pso(X_train, X_train_original, mean_X, std_X, ...
        params, relu, optimizers{o}, num_epochs, o);
    string = sprintf("personal bests (first 5): %d, %d, %d, %d, %d", p_best_loss(1), p_best_loss(2), p_best_loss(3), p_best_loss(4), p_best_loss(5));
    disp(string)
    weights_log(o).optimizer = tmp_log.optimizer;
    weights_log(o).epoch = tmp_log.epoch;
    loss_all(:, o) = loss_history;
end



