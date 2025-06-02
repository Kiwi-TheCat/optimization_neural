

addpath('training');
%% Step 1: Load the best weight matrix of Adam method
clear; clc;
load("Adam_weights_and_bias.mat");
load("data_1s.mat")

% Access original weights and biases from a specific epoch
original_params.We1        = params.We1;
original_params.We_latent  = params.We_latent;
original_params.Wd1        = params.Wd1;
original_params.Wd_output  = params.Wd_output;
original_params.be1        = params.be1;
original_params.be_latent  = params.be_latent;
original_params.bd1        = params.bd1;
original_params.bd_output  = params.bd_output;

%% Step 2: Add noise layer by layer
data = data(1:end-1,1:end-1);
delta = diff(data);
X_train_original = delta(1:384, :);
X_train_original = double(X_train_original);

% Noise sensitivity setup
Noise_Levels = [0.001, 0.01, 0.1]; % small noise, moderate noise, large noise
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];
num_samples = 500;

% Initialize table data
results = table('Size', [0 3], ...
    'VariableTypes', {'string', 'double', 'double'}, ...
    'VariableNames', {'LayerName', 'Noise', 'Loss'});

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

        % Calculate mean and std for normalization (as used in training)
        mean_X = mean(X_train_original, 1);
        std_X = std(X_train_original, 0, 1);
        std_X(std_X == 0) = 1; % Avoid division by zero
        
        % Evaluate the loss using existing function
        current_loss = compute_reconstruction_mse(temp_params, X_train_original, mean_X, std_X, @relu);

        % Append to table
        results = [results; {current_layer, current_noise, current_loss}];
    end
end

%% Display results
disp(results);

%% Compare With Original Loss

alpha = 0.05; % for 95% confidence interval

% Calculate baseline loss using existing function
mean_X = mean(X_train_original, 1);
std_X = std(X_train_original, 0, 1);
std_X(std_X == 0) = 1; % Avoid division by zero

baseline_loss = compute_reconstruction_mse(original_params, X_train_original, mean_X, std_X, @relu);

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
            noisy_W = W + current_noise * randn(size(W));
            temp_params = original_params;
            temp_params.(current_layer) = noisy_W;
            
            % Calculate normalization parameters
            mean_X = mean(X_train_original, 1);
            std_X = std(X_train_original, 0, 1);
            std_X(std_X == 0) = 1; % Avoid division by zero
            
            losses(k) = compute_reconstruction_mse(temp_params, X_train_original, mean_X, std_X, @relu);
        end

        avg_loss = mean(losses);
        delta_loss = avg_loss - baseline_loss;
        std_dev = std(losses);
        % 95% CI using t-distribution
        t_multiplier = tinv(1 - alpha/2, num_samples - 1);
        conf_interval = t_multiplier * std_dev / sqrt(num_samples);

        % Append to results table
        results = [results; {current_layer, current_noise, avg_loss, delta_loss, std_dev, conf_interval}];

        % Log the individual losses
        field_name = sprintf('%s_%.4f', current_layer, current_noise);
        field_name = strrep(field_name, '.', 'p');  % replace '.' with 'p'
        loss_log.(field_name) = losses;
    end
end

disp(results);

%% Bar plot of Layer sensitivity

% Extract unique layer names and noise levels
layers = unique(results.LayerName, 'stable');
Noise_Levels = unique(results.Noise);

num_layers = length(layers);
num_noises = length(Noise_Levels);

% Preallocate
delta_matrix = nan(num_layers, num_noises);
ci_matrix = nan(num_layers, num_noises);

% Fill the matrices
for i = 1:num_layers
    for j = 1:num_noises
        idx = strcmp(results.LayerName, layers(i)) & abs(results.Noise - Noise_Levels(j)) < 1e-10;
        if any(idx)
            delta_matrix(i, j) = results.DeltaLoss(find(idx, 1));
            ci_matrix(i, j) = results.ConfInterval(find(idx, 1));
        end
    end
end

% Check if we have positive values for log scale
% Remove NaN values first, then check if all remaining values are positive
valid_values = delta_matrix(~isnan(delta_matrix));
has_positive_values = ~isempty(valid_values) && all(valid_values > 0);

figure;
hold on;

b = bar(delta_matrix, 'grouped');

% Set X-tick labels
set(gca, 'XTickLabel', layers);
xlabel('Layer');
ylabel('ΔLoss');
title('Layer Sensitivity with Confidence Intervals');

% Only use log scale if all values are positive
if has_positive_values
    set(gca, 'YScale', 'log');
    ylabel('ΔLoss (log scale)');
    title('Layer Sensitivity with Confidence Intervals (log scale)');
else
    % Use linear scale and warn user
    fprintf('Warning: Some delta loss values are negative or zero. Using linear scale.\n');
end

grid on;

% Error bars
groupwidth = min(0.8, num_noises / (num_noises + 1.5));
for j = 1:num_noises
    x = (1:num_layers) - groupwidth/2 + (2*j-1) * groupwidth / (2*num_noises);
    errorbar(x, delta_matrix(:, j), ci_matrix(:, j), 'k.', ...
             'LineStyle', 'none', 'CapSize', 6);
end

% Legend: force consistent noise level colors
colors = lines(num_noises);
for j = 1:num_noises
    b(j).FaceColor = colors(j, :);
end
legend(arrayfun(@(x) sprintf('Noise = %.4f', x), Noise_Levels, 'UniformOutput', false), ...
       'Location', 'northwest');

hold off;

%% Box plot of Layer sensitivity (FIXED VERSION)

% Define layer and noise info
layers = ["We1", "We_latent", "Wd1", "Wd_output"];
noise_levels = [0.001, 0.01]; % Only using the two smaller noise levels

% Flatten loss_log into Arrays
all_losses = [];
group_labels = [];

for i = 1:length(layers)
    for j = 1:length(noise_levels)
        layer = layers(i);
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

% Color the boxes
h = findobj(gca, 'Tag', 'Box');
colors = lines(numel(h));
for j = 1:numel(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), colors(j,:), 'FaceAlpha', .5);
end

%% Heatmap of ΔLoss per Layer per Noise

layers = unique(results.LayerName, 'stable');
Noise_Levels = unique(results.Noise);

num_layers = length(layers);
num_noises = length(Noise_Levels);

% Preallocate the matrix (noise levels are rows, layers are columns)
delta_matrix = nan(num_noises, num_layers);

for i = 1:num_layers
    for j = 1:num_noises
        idx = strcmp(results.LayerName, layers(i)) & abs(results.Noise - Noise_Levels(j)) < 1e-6;
        if any(idx)
            delta_matrix(j, i) = results.DeltaLoss(find(idx, 1));
        end
    end
end

figure;
heatmap(string(layers), ...
        arrayfun(@(x) sprintf('%.4f', x), Noise_Levels, 'UniformOutput', false), ...
        delta_matrix, ...
        'XLabel', 'Layer', ...
        'YLabel', 'Noise Level', ...
        'Title', 'ΔLoss Heatmap: Sensitivity by Layer and Noise', ...
        'ColorbarVisible', 'on');

%% Supporting functions
function y = relu(x)
    y = max(0, x);
end