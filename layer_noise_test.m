%% Step 1: Load the best weight matrix of Adam method
clear; clc;
load("Adam_weights_and_bias.mat");
load("data_1s.mat")

% Access original weights from a specific epoch
original_weights.We1        = params.We1;
original_weights.We_latent  = params.We_latent;
original_weights.Wd1        = params.Wd1;
original_weights.Wd_output  = params.Wd_output;
original_bias.be1           = params.be1;
original_bias.be_latent     = params.be_latent;
original_bias.bd1           = params.bd1;
original_bias.bd_output     = params.bd_output;


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
        W = original_weights.(current_layer);

        % Add Gaussian noise
        noisy_W = W + current_noise * randn(size(W));

        % Inject the noisy weights back into the model
        temp_weights = original_weights;
        temp_weights.(current_layer) = noisy_W;

        % Evaluate the loss (you need a custom function here)
        current_loss = evaluate_model_loss(temp_weights,original_bias,X_train_original); % Implement this!

        % Append to table
        results = [results; {current_layer, current_noise, current_loss}];
    end
end

%% Display results
disp(results);

%% Compare With Original Loss

alpha = 0.05; % for 95% confidence interval

baseline_loss = evaluate_model_loss(original_weights, original_bias, X_train_original);

% Initialize results table with extra columns
results = table('Size', [0 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'LayerName', 'Noise', 'AvgLoss', 'DeltaLoss', 'StdDev', 'ConfInterval'});

% Create a structure to log all 10 losses
loss_log = struct();

for i = 1:length(Layer_Names)
    for j = 1:length(Noise_Levels)
        current_layer = Layer_Names(i);
        current_noise = Noise_Levels(j);
        W = original_weights.(current_layer);
        
        losses = zeros(1, num_samples);
        
        for k = 1:num_samples
            noisy_W = W + current_noise * randn(size(W));
            temp_weights = original_weights;
            temp_weights.(current_layer) = noisy_W;
            
            losses(k) = evaluate_model_loss(temp_weights, original_bias, X_train_original);
        end

        avg_loss = mean(losses);
        delta_loss = avg_loss - baseline_loss;
        std_dev = std(losses);
        % 95% CI using t-distribution (n = 10 → df = 9)
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
Noise_Levels = unique(results.Noise);  % Use your defined Noise_Levels here

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

layers = unique(results.LayerName, 'stable');
Noise_Levels = unique(results.Noise);
num_layers = length(layers);
num_noises = length(Noise_Levels);

figure;
hold on;

b = bar(delta_matrix, 'grouped');

% Set X-tick labels
set(gca, 'XTickLabel', layers);
xlabel('Layer');
ylabel('ΔLoss (log scale)');
title('Layer Sensitivity with Confidence Intervals (log scale)');
set(gca, 'YScale', 'log');  % ✅ log scale here
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

%% Box plot of Layer sensitivity

% step1: define layer and noise info
layers = ["We1", "We_latent", "Wd1", "Wd_output"];
noise_levels = [0.01, 0.001];

% step2: flatten loss_log into Arrays
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

% step 3: create the box plot
figure;
boxplot(all_losses, group_labels);
xlabel('Layer + Noise Level');
ylabel('Loss');
title('Distribution of Loss with Gaussian Noise (500 samples)');
grid on;

h = findobj(gca, 'Tag', 'Box');
colors = lines(numel(h));
for j = 1:numel(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), colors(j,:), 'FaceAlpha', .5);
end

%% Heatmap of ΔLoss per Layer per Noise

layers = unique(results.LayerName, 'stable');
Noise_Levels = unique(results.Noise); % same case as defined earlier

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



%% Supporting function
function y = relu(x)
    y = max(0, x);
end

function loss = evaluate_model_loss(weights,bias,X_train_original)
mean_X = mean(X_train_original, 1); % Calculate mean for each feature (column)
std_X = std(X_train_original, 0, 1);  % Calculate std for each feature (column)

% Use your existing test data and parameters from main2.m
test_sample_normalized = (X_train_original(1, :) - mean_X) ./ std_X;

% Forward pass with the weights
H1_enc_linear = test_sample_normalized * weights.We1 + bias.be1;
H1_enc_activated = relu(H1_enc_linear);

Z_linear = H1_enc_activated * weights.We_latent + bias.be_latent;
Z_activated = relu(Z_linear);

H1_dec_linear = Z_activated * weights.Wd1 + bias.bd1;
H1_dec_activated = relu(H1_dec_linear);

X_reconstructed = H1_dec_activated * weights.Wd_output + bias.bd_output;

% Calculate loss
reconstruction_error = X_reconstructed - test_sample_normalized;
loss = 0.5 * sum(reconstruction_error.^2);
end




