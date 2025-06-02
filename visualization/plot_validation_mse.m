% Load from one folder above the current directory
load('../training/weights_log_tensor_leaky.mat');
load('../training/loss_all_log_leaky.mat');
load('../training/preprocessed_full_data.mat', 'X', 'X_original', 'mean_X', 'std_X');
addpath(fullfile('..', 'training'));

% --- Setup ---

num_samples = size(X,1);
num_samples_train = ceil(num_samples*0.8);
X_val = X(num_samples_train:end,:);
num_optimizers = numel(weights_log);

all_train_loss = cell(num_optimizers, 1);
all_val_loss   = cell(num_optimizers, 1);
mean_losses    = zeros(num_optimizers, 2);  % [train, val]

[~, ~, relu, ~, ~, ~] = setup_network();  % or leaky_relu if used

% --- Evaluate each optimizer ---
batch_size = 100;

for o = 1:num_optimizers
    params = weights_log(o).epoch(end);

    % --- Training loss ---
    N_train = size(X_train, 1);
    num_batches_train = ceil(N_train / batch_size);
    loss_train = zeros(N_train, 1);

    for b = 1:num_batches_train
        idx_start = (b - 1) * batch_size + 1;
        idx_end   = min(b * batch_size, N_train);
        X_batch = X_train(idx_start:idx_end, :);

        batch_mse = compute_reconstruction_mse(params, X_batch, relu);
        loss_train(idx_start:idx_end) = batch_mse;  % could be scalar or vector
    end

    % --- Validation loss ---
    N_val = size(X_val, 1);
    num_batches_val = ceil(N_val / batch_size);
    loss_val = zeros(N_val, 1);

    for b = 1:num_batches_val
        idx_start = (b - 1) * batch_size + 1;
        idx_end   = min(b * batch_size, N_val);
        X_batch = X_val(idx_start:idx_end, :);

        batch_mse = compute_reconstruction_mse(params, X_batch, relu);
        loss_val(idx_start:idx_end) = batch_mse;
    end

    % Store
    all_train_loss{o} = loss_train;
    all_val_loss{o}   = loss_val;
    mean_losses(o, :) = [mean(loss_train), mean(loss_val)];
end


%% --- Boxplot ---
all_losses = vertcat(all_train_loss{:}, all_val_loss{:});
group_labels = {};

for o = 1:num_optimizers
    group_labels = [group_labels; ...
        repmat({[upper(weights_log(o).optimizer) ' - Train']}, size(all_train_loss{o})); ...
        repmat({[upper(weights_log(o).optimizer) ' - Val']},   size(all_val_loss{o}))];
end

figure('Name', 'Boxplot of Reconstruction Losses per Optimizer');
boxplot(all_losses, group_labels);
ylabel('Reconstruction Loss');
title('Boxplot: Train vs Validation Loss per Optimizer');
xtickangle(30); grid on;

%% --- Barplot ---
figure('Name', 'Barplot of Mean Reconstruction Losses');
bar(mean_losses);
set(gca, 'XTickLabel', upper({weights_log.optimizer}));
legend({'Train', 'Validation'});
ylabel('Mean Reconstruction Loss');
title('Mean Loss: Train vs Validation per Optimizer');
grid on;

