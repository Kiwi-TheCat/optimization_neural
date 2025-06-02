% Load from one folder above the current directory
load('../training/weights_log_tensor_3.mat');
load('../training/loss_all_log_3.mat');
load('../training/preprocessed_full_data.mat', 'X_train', 'X_original', 'mean_X', 'std_X');
addpath(fullfile('..', 'training'));

% --- Setup ---
X_val   = X_train(385:770, :);  % validation
X_train = X_train(1:384, :);    % training
num_optimizers = numel(weights_log);

all_train_loss = cell(num_optimizers, 1);
all_val_loss   = cell(num_optimizers, 1);
mean_losses    = zeros(num_optimizers, 2);  % [train, val]

[~, ~, relu, ~, ~, ~] = setup_network();  % or leaky_relu if used

% --- Evaluate each optimizer ---
for o = 1:num_optimizers
    params = weights_log(o).epoch(end);  % final epoch weights
    
    % Compute training losses
    loss_train = zeros(size(X_train, 1), 1);
    for i = 1:size(X_train, 1)
        loss_train(i) = compute_reconstruction_mse(params, X_train(i,:), relu);
    end

    % Compute validation losses
    loss_val = zeros(size(X_val, 1), 1);
    for i = 1:size(X_val, 1)
        loss_val(i) = compute_reconstruction_mse(params, X_val(i,:), relu);
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

