% Simplified Autoencoder Training with Modular Optimizer Support
clear; clc; close all
load('data_1s.mat')

%% Parameters
n = 10;
chunk_size = 384;
n_channel = 384;
raw = data(1:n*chunk_size, 1:n_channel);
delta = diff(raw);

%% Select training data
num_train_samples = 384;
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
num_epochs = 50;
learning_rate = 0.0002;
optimizers = {'sgd', 'adagrad', 'adam'};

%% Store comparison results
weights_log = struct();
loss_all = zeros(num_epochs, numel(optimizers));
mse_all = zeros(1, numel(optimizers));
x_train_log = cell(num_epochs, numel(optimizers));
x_test_log = cell(num_epochs, numel(optimizers));

for o = 1:numel(optimizers)
    optimizer_type = optimizers{o};
    fprintf('Training with %s optimizer...\n', optimizer_type);

    [params, optim, relu, relu_deriv] = setup_network(input_size, hidden_size, latent_size);

    [loss_history, tmp_log, mse_all(o), x_test_log(:,o)] = train_autoencoder(X_train, X_train_original, mean_X, std_X, ...
        params, optim, relu, relu_deriv, optimizer_type, learning_rate, num_epochs, o);

    weights_log(o).optimizer = tmp_log.optimizer;
    weights_log(o).epoch = tmp_log.epoch;
    loss_all(:, o) = loss_history;
end
save('loss_all_log.mat', 'loss_all', 'mse_all');
save('weights_log_tensor.mat', 'weights_log', 'x_test_log');


