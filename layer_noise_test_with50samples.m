% Author: Kunxin Wu
% Date: 2nd June 2025
%% Step 1: Load the best weight matrix of Adam method
clear; clc;
load("Adam_weights_and_bias.mat");
load("./training/preprocessed_full_data.mat")
addpath(fullfile('.', 'training')); % like this the functions from training become usable

% Access original weights and biases from a specific epoch
X_train = X_train(1:384, 1:384);  % first 384 samples, all columns(:= full signal)
original_params = params;
alpha_leaky = 0.1;
[~, ~, relu, leaky_relu, ~, ~] = setup_network(0, 0, 0, alpha_leaky); 

%% Step 2: Add noise layer by layer
% Noise sensitivity setup
Noise_Levels = [0.001, 0.01, 0.1]; % small noise, moderate noise, large noise
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];
num_samples = 500;

% Initialize table data
results = table('Size', [0 3], ...
    'VariableTypes', {'string', 'double', 'double'}, ...
    'VariableNames', {'LayerName', 'Noise', 'Loss'});

baseline_loss = compute_reconstruction_mse(original_params, X_train, relu);

% Loop through each layer and noise level
for i = 1:length(Layer_Names)
    for j = 1:length(Noise_Levels)
        current_layer = Layer_Names(i);
        current_noise = Noise_Levels(j);

        % Get the original weights for the current layer
        W = original_params.(current_layer);

        % Add Gaussian noise
        noisy_W = W + current_noise * randn(size(W));

        % Inject the noisy weights back into the model
        temp_params = original_params;
        temp_params.(current_layer) = noisy_W;

        % Evaluate the loss using enhanced batch function
        current_loss = compute_reconstruction_mse(temp_params, X_train, relu);

        % Append to table
        results = [results; {current_layer, current_noise, current_loss}];
    end
end

% Display results
disp(results);

% Compare With Original Loss
alpha = 0.05; % for 95% confidence interval


% Initialize results table with extra columns
results = table('Size', [0 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'LayerName', 'Noise', 'AvgLoss', 'DeltaLoss', 'StdDev', 'ConfInterval'});

% Create a structure to log all losses
loss_log = struct();

for i = 1:length(Layer_Names)
    for j = 1:length(Noise_Levels)
        current_layer = Layer_Names(i);
        current_noise = Noise_Levels(j);
        W = original_params.(current_layer);

        losses = zeros(1, num_samples);
        for k = 1:num_samples
            temp_params = original_params;
            temp_params.(current_layer) = W + current_noise * randn(size(W));
            losses(k) = compute_reconstruction_mse(temp_params, X_train, relu);
        end

        [avg_loss, delta_loss, std_dev, conf_interval] = analyze_losses(losses, baseline_loss, alpha);

        % Log result
        results = [results; {current_layer, current_noise, avg_loss, delta_loss, std_dev, conf_interval}];

        % Store raw losses
        key = sprintf('%s_%.4f', current_layer, current_noise);
        key = strrep(key, '.', 'p');
        loss_log.(key) = losses;
    end
end


disp(results);
%% === Display the results ===
% === Box plot of Layer sensitivity ===

% Define layer and noise info
Layer_Names  = ["We1", "We_latent", "Wd1", "Wd_output"];
noise_levels = [0.001, 0.01]; % better for visualization

% Flatten loss_log into Arrays
all_losses = [];
group_labels = [];

for i = 1:length(Layer_Names )
    for j = 1:length(noise_levels)
        layer = Layer_Names (i);
        noise = noise_levels(j);

        % Construct the field name used in the struct
        field_name = sprintf('%s_%.4f', layer, noise);
        field_name = strrep(field_name, '.', 'p');

        if isfield(loss_log, field_name)
            losses = loss_log.(field_name);
            n = length(losses);
            label = sprintf('%s (σ=%.3f)', layer, noise);
            all_losses = [all_losses; losses(:)];
            group_labels = [group_labels; repmat({label}, n, 1)];
        end
    end
end

% Check for negative values and handle appropriately
min_loss = min(all_losses);
if min_loss <= 0
    fprintf('Warning: Found negative or zero loss values (min = %.6f). Adding offset for visualization.\n', min_loss);
    % Add small offset to make all values positive if needed for log operations
    all_losses_adjusted = all_losses - min_loss + 1e-10;
else
    all_losses_adjusted = all_losses;
end

% Create the box plot
figure;
boxplot(all_losses_adjusted, group_labels);
xlabel('Layer + Noise Level');
ylabel('Loss');
title('Distribution of Loss with Gaussian Noise (500 samples)');
grid on;

% Add horizontal line for baseline loss
hold on;
baseline_line = yline(baseline_loss, '--r', 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2]);
baseline_line.Label = sprintf('Baseline Loss = %.2f', baseline_loss);
baseline_line.LabelHorizontalAlignment = 'left';
baseline_line.LabelVerticalAlignment = 'bottom';

% Color the boxes
h = findobj(gca, 'Tag', 'Box');
colors = lines(numel(h));
for j = 1:numel(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), colors(j,:), 'FaceAlpha', .5);
end

hold off;

% === Heatmap of ΔLoss per Layer per Noise ===

num_layers = length(Layer_Names );
num_noises = length(Noise_Levels);

% Preallocate the matrix (noise levels are rows, layers are columns)
delta_matrix = nan(num_noises, num_layers);

for i = 1:num_layers
    for j = 1:num_noises
        idx = strcmp(results.LayerName, Layer_Names(i)) & abs(results.Noise - Noise_Levels(j)) < 1e-6;
        if any(idx)
            delta_matrix(j, i) = results.DeltaLoss(find(idx, 1));
        end
    end
end
% Escape underscores for display
escaped_layers = strrep(string(Layer_Names), '_', '\_');

figure;
heatmap(escaped_layers, ...
        arrayfun(@(x) sprintf('%.4f', x), Noise_Levels, 'UniformOutput', false), ...
        delta_matrix, ...
        'XLabel', 'Layer', ...
        'YLabel', 'Noise Level', ...
        'Title', 'ΔLoss Heatmap: Sensitivity by Layer and Noise', ...
        'ColorbarVisible', 'on');


%% helper functions
% function noisy_params = add_noise_to_layer(params, layer, noise_std)
%     noisy_params = params;
%     W = params.(layer);
%     noisy_params.(layer) = W + noise_std * randn(size(W));
% end

function [mean_val, delta, std_val, conf] = analyze_losses(losses, baseline, alpha)
    mean_val = mean(losses);
    delta = mean_val - baseline;
    std_val = std(losses);
    t = tinv(1 - alpha / 2, numel(losses) - 1);
    conf = t * std_val / sqrt(numel(losses));
end
