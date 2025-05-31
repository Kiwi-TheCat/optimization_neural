%% Enhanced Convergence Sensitivity Analysis with Flatness Metrics

clear; clc;

% Load the training results from noise analysis
try
    load('enhanced_trained_noisy_weights.mat');
    fprintf('Loaded training results from enhanced_trained_noisy_weights.mat\n');
catch
    fprintf('Error: enhanced_trained_noisy_weights.mat not found!\n');
    return;
end

% Load original data for validation
load("Adam_weights_and_bias.mat");
load("data_1s.mat");

% Preprocess data (same as training)
data = data(1:end-1,1:end-1);
delta = diff(data);
X_train_original = delta(1:384, :);
X_train_original = double(X_train_original);
mean_X = mean(X_train_original, 1);
std_X = std(X_train_original, 0, 1);
std_X(std_X == 0) = 1e-6;
X_train = (X_train_original - mean_X) ./ std_X;

% Use subset for flatness analysis (computational efficiency)
X_test = X_train(1:100, :); % Use first 100 samples for flatness tests

fprintf('=== ENHANCED CONVERGENCE & FLATNESS ANALYSIS ===\n');

%% 1. Extract and organize existing training data
unique_layers = unique(results.LayerName);
unique_noise_levels = unique(results.NoiseLevel);
unique_noise_levels = unique_noise_levels(unique_noise_levels > 0);

fprintf('Found data for %d layers and %d noise levels\n', length(unique_layers), length(unique_noise_levels));

% Get original baseline
original_data = results(results.NoiseLevel == 0, :);
if isempty(original_data)
    original_loss = evaluate_loss(original_params, X_test);
    fprintf('Baseline loss: %.6f\n', original_loss);
else
    original_loss = original_data.FinalLoss(1);
    fprintf('Baseline loss: %.6f\n', original_loss);
end

%% 2. FLATNESS ANALYSIS - Key Methods Only
fprintf('\n=== FLATNESS ANALYSIS ===\n');

% Configuration for flatness analysis
flatness_config = struct();
flatness_config.perturbation_scales = [0.001, 0.01, 0.05]; % Reduced for efficiency
flatness_config.num_random_directions = 20; % Reduced for efficiency
flatness_config.sharpness_radius = 0.02; % SAM-inspired radius

% Initialize flatness results storage
flatness_results = table('Size', [0 8], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'LayerName', 'NoiseLevel', 'FlatnessScore', 'Sharpness', 'LossVariance', ...
                      'AvgPerturbSensitivity', 'MaxPerturbSensitivity', 'FinalLoss'});

%% 3. Analyze each trained model for flatness
experiment_idx = 1;
fprintf('Analyzing flatness for each experiment...\n');

for i = 1:height(results)
    if results.NoiseLevel(i) == 0
        continue; % Skip original
    end
    
    layer_name = results.LayerName{i};
    noise_level = results.NoiseLevel(i);
    final_loss = results.FinalLoss(i);
    
    fprintf('\nAnalyzing %s with noise %.3f...\n', layer_name, noise_level);
    
    % Get final parameters for this experiment
    if experiment_idx <= length(final_params_all) && ~isempty(final_params_all{experiment_idx})
        params = final_params_all{experiment_idx};
        
        % 1. Random Perturbation Flatness (most important)
        [flatness_score, avg_sens, max_sens] = compute_perturbation_flatness(params, X_test, flatness_config);
        
        % 2. Sharpness Analysis (SAM-inspired)
        sharpness = compute_sharpness_metric(params, X_test, flatness_config.sharpness_radius);
        
        % 3. Loss Variance around minimum
        loss_variance = compute_loss_variance(params, X_test);
        
        % Store results
        flatness_results = [flatness_results; {layer_name, noise_level, flatness_score, sharpness, ...
                           loss_variance, avg_sens, max_sens, final_loss}];
        
        fprintf('  Flatness Score: %.6f (lower = flatter)\n', flatness_score);
        fprintf('  Sharpness: %.6f\n', sharpness);
        fprintf('  Loss Variance: %.6f\n', loss_variance);
        
    else
        fprintf('  No parameters available for analysis\n');
    end
    
    experiment_idx = experiment_idx + 1;
end

%% 4. Noise Impact Analysis
fprintf('\n=== NOISE IMPACT ON FLATNESS ===\n');

% Analyze how noise level affects flatness
for layer = unique_layers'
    layer_data = flatness_results(strcmp(flatness_results.LayerName, layer), :);
    if height(layer_data) > 1
        fprintf('\n%s Layer - Noise Impact:\n', layer{1});
        
        % Sort by noise level
        [sorted_noise, idx] = sort(layer_data.NoiseLevel);
        sorted_flatness = layer_data.FlatnessScore(idx);
        sorted_sharpness = layer_data.Sharpness(idx);
        sorted_variance = layer_data.LossVariance(idx);
        
        % Analyze trends
        flatness_trend = analyze_trend(sorted_noise, sorted_flatness);
        sharpness_trend = analyze_trend(sorted_noise, sorted_sharpness);
        variance_trend = analyze_trend(sorted_noise, sorted_variance);
        
        fprintf('  Flatness trend: %s\n', flatness_trend);
        fprintf('  Sharpness trend: %s\n', sharpness_trend);
        fprintf('  Variance trend: %s\n', variance_trend);
        
        % Overall assessment
        if contains(flatness_trend, 'decreasing') && contains(sharpness_trend, 'decreasing')
            fprintf('  → Noise IMPROVES flatness (flatter minima)\n');
        elseif contains(flatness_trend, 'increasing') && contains(sharpness_trend, 'increasing')
            fprintf('  → Noise REDUCES flatness (sharper minima)\n');
        else
            fprintf('  → Mixed effects of noise on flatness\n');
        end
    end
end

%% 5. Consolidated Visualization
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Flatness Score vs Noise Level
subplot(2, 3, 1);
for layer = unique_layers'
    layer_data = flatness_results(strcmp(flatness_results.LayerName, layer), :);
    if ~isempty(layer_data)
        semilogx(layer_data.NoiseLevel, layer_data.FlatnessScore, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', layer{1});
        hold on;
    end
end
xlabel('Noise Level');
ylabel('Flatness Score (lower = flatter)');
title('Flatness vs Noise Level');
legend('Location', 'best');
grid on;

% Plot 2: Sharpness vs Noise Level
subplot(2, 3, 2);
for layer = unique_layers'
    layer_data = flatness_results(strcmp(flatness_results.LayerName, layer), :);
    if ~isempty(layer_data)
        semilogx(layer_data.NoiseLevel, layer_data.Sharpness, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', layer{1});
        hold on;
    end
end
xlabel('Noise Level');
ylabel('Sharpness');
title('Sharpness vs Noise Level');
legend('Location', 'best');
grid on;

% Plot 3: Loss Variance vs Noise Level
subplot(2, 3, 3);
for layer = unique_layers'
    layer_data = flatness_results(strcmp(flatness_results.LayerName, layer), :);
    if ~isempty(layer_data)
        semilogx(layer_data.NoiseLevel, layer_data.LossVariance, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', layer{1});
        hold on;
    end
end
xlabel('Noise Level');
ylabel('Loss Variance');
title('Loss Variance vs Noise Level');
legend('Location', 'best');
grid on;

% Plot 4: Flatness vs Performance Scatter
subplot(2, 3, 4);
scatter(flatness_results.FlatnessScore, flatness_results.FinalLoss, 100, flatness_results.NoiseLevel, 'filled');
xlabel('Flatness Score');
ylabel('Final Loss');
title('Flatness vs Performance');
colorbar;
colormap('parula');
c = colorbar;
c.Label.String = 'Noise Level';
grid on;

% Plot 5: Multi-metric Flatness Comparison
subplot(2, 3, 5);
% Normalize metrics for comparison
norm_flatness = (flatness_results.FlatnessScore - min(flatness_results.FlatnessScore)) / ...
                (max(flatness_results.FlatnessScore) - min(flatness_results.FlatnessScore));
norm_sharpness = (flatness_results.Sharpness - min(flatness_results.Sharpness)) / ...
                 (max(flatness_results.Sharpness) - min(flatness_results.Sharpness));
norm_variance = (flatness_results.LossVariance - min(flatness_results.LossVariance)) / ...
                (max(flatness_results.LossVariance) - min(flatness_results.LossVariance));

plot(flatness_results.NoiseLevel, norm_flatness, 'o-', 'LineWidth', 2, 'DisplayName', 'Flatness Score');
hold on;
plot(flatness_results.NoiseLevel, norm_sharpness, 's-', 'LineWidth', 2, 'DisplayName', 'Sharpness');
plot(flatness_results.NoiseLevel, norm_variance, '^-', 'LineWidth', 2, 'DisplayName', 'Loss Variance');
set(gca, 'XScale', 'log');
xlabel('Noise Level');
ylabel('Normalized Metric Value');
title('Normalized Flatness Metrics');
legend('Location', 'best');
grid on;

% Plot 6: Summary Statistics
subplot(2, 3, 6);
summary_stats = zeros(length(unique_layers), 4);
for i = 1:length(unique_layers)
    layer_data = flatness_results(strcmp(flatness_results.LayerName, unique_layers{i}), :);
    if ~isempty(layer_data)
        summary_stats(i, 1) = mean(layer_data.FlatnessScore);
        summary_stats(i, 2) = mean(layer_data.Sharpness) * 100; % Scale for visualization
        summary_stats(i, 3) = mean(layer_data.LossVariance) * 10; % Scale for visualization
        summary_stats(i, 4) = mean(layer_data.FinalLoss);
    end
end

bar(summary_stats);
set(gca, 'XTickLabel', unique_layers);
ylabel('Average Metric Value');
title('Average Flatness Metrics by Layer');
legend({'Flatness Score', 'Sharpness (×100)', 'Loss Variance (×10)', 'Final Loss'}, 'Location', 'best');
grid on;

sgtitle('Enhanced Convergence & Flatness Analysis: How Noise Affects Minima');

%% 6. Comprehensive Summary Report
fprintf('\n=== COMPREHENSIVE SUMMARY REPORT ===\n');

% Overall noise impact assessment
all_flatness_corr = corr(flatness_results.NoiseLevel, flatness_results.FlatnessScore);
all_sharpness_corr = corr(flatness_results.NoiseLevel, flatness_results.Sharpness);
all_variance_corr = corr(flatness_results.NoiseLevel, flatness_results.LossVariance);

fprintf('\nOverall Noise Impact (correlation coefficients):\n');
fprintf('Noise vs Flatness Score: %.4f %s\n', all_flatness_corr, interpret_correlation(all_flatness_corr, 'flatness'));
fprintf('Noise vs Sharpness: %.4f %s\n', all_sharpness_corr, interpret_correlation(all_sharpness_corr, 'sharpness'));
fprintf('Noise vs Loss Variance: %.4f %s\n', all_variance_corr, interpret_correlation(all_variance_corr, 'variance'));

% Key findings
fprintf('\nKey Findings:\n');
avg_flatness_improvement = (max(flatness_results.FlatnessScore) - min(flatness_results.FlatnessScore)) / max(flatness_results.FlatnessScore) * 100;
avg_sharpness_improvement = (max(flatness_results.Sharpness) - min(flatness_results.Sharpness)) / max(flatness_results.Sharpness) * 100;

fprintf('1. Flatness Score Range: %.1f%% improvement from lowest to highest noise\n', avg_flatness_improvement);
fprintf('2. Sharpness Range: %.1f%% improvement from lowest to highest noise\n', avg_sharpness_improvement);

% Best performing configuration
[~, best_idx] = min(flatness_results.FlatnessScore);
best_config = flatness_results(best_idx, :);
fprintf('3. Flattest minimum: %s layer with %.3f noise (score: %.6f)\n', ...
    best_config.LayerName{1}, best_config.NoiseLevel, best_config.FlatnessScore);

% Noise threshold analysis
low_noise_data = flatness_results(flatness_results.NoiseLevel <= 0.01, :);
high_noise_data = flatness_results(flatness_results.NoiseLevel > 0.01, :);

if ~isempty(low_noise_data) && ~isempty(high_noise_data)
    low_noise_flatness = mean(low_noise_data.FlatnessScore);
    high_noise_flatness = mean(high_noise_data.FlatnessScore);
    
    fprintf('4. Low noise (≤0.01) avg flatness: %.6f\n', low_noise_flatness);
    fprintf('   High noise (>0.01) avg flatness: %.6f\n', high_noise_flatness);
    
    if high_noise_flatness < low_noise_flatness
        fprintf('   → Higher noise leads to FLATTER minima\n');
    else
        fprintf('   → Higher noise leads to SHARPER minima\n');
    end
end

fprintf('\nConclusion: ');
if all_flatness_corr < -0.3 && all_sharpness_corr < -0.3
    fprintf('Noise injection SIGNIFICANTLY IMPROVES flatness (creates flatter minima)\n');
elseif all_flatness_corr < 0 && all_sharpness_corr < 0
    fprintf('Noise injection MODERATELY IMPROVES flatness\n');
else
    fprintf('Noise injection has MIXED EFFECTS on flatness\n');
end

%% Supporting Functions for Flatness Analysis

function [flatness_score, avg_sensitivity, max_sensitivity] = compute_perturbation_flatness(params, X_test, config)
    % Compute flatness based on random perturbations
    
    base_loss = evaluate_model_loss(params, X_test);
    scales = config.perturbation_scales;
    num_directions = config.num_random_directions;
    
    sensitivities = [];
    
    for scale = scales
        scale_sensitivities = [];
        for dir = 1:num_directions
            % Generate random perturbation
            perturbed_params = generate_random_perturbation(params, scale);
            perturbed_loss = evaluate_model_loss(perturbed_params, X_test);
            
            % Compute sensitivity (relative loss change)
            sensitivity = abs(perturbed_loss - base_loss) / (base_loss + 1e-8);
            scale_sensitivities = [scale_sensitivities; sensitivity];
        end
        sensitivities = [sensitivities; scale_sensitivities];
    end
    
    % Compute flatness metrics
    avg_sensitivity = mean(sensitivities);
    max_sensitivity = max(sensitivities);
    flatness_score = avg_sensitivity; % Lower = flatter
end

function sharpness = compute_sharpness_metric(params, X_test, radius)
    % SAM-inspired sharpness computation
    
    base_loss = evaluate_model_loss(params, X_test);
    
    % Generate adversarial perturbation direction
    gradients = compute_finite_diff_gradients(params, X_test);
    grad_norm = compute_gradient_norm(gradients);
    
    if grad_norm > 1e-8
        % Create unit perturbation in gradient direction
        perturbed_params = params;
        param_names = fieldnames(params);
        
        for i = 1:length(param_names)
            name = param_names{i};
            perturbed_params.(name) = params.(name) + radius * gradients.(name) / grad_norm;
        end
        
        perturbed_loss = evaluate_model_loss(perturbed_params, X_test);
        sharpness = max(0, perturbed_loss - base_loss) / (base_loss + 1e-8);
    else
        sharpness = 0;
    end
end

function variance = compute_loss_variance(params, X_test)
    % Compute loss variance with small Gaussian perturbations
    
    num_samples = 50;
    sigma = 0.01;
    losses = zeros(num_samples, 1);
    
    for i = 1:num_samples
        % Sample from Gaussian around parameters
        noisy_params = sample_gaussian_params(params, sigma);
        losses(i) = evaluate_model_loss(noisy_params, X_test);
    end
    
    variance = var(losses);
end

function trend_desc = analyze_trend(x_values, y_values)
    % Analyze if trend is increasing, decreasing, or stable
    
    if length(x_values) < 2
        trend_desc = 'insufficient data';
        return;
    end
    
    % Simple linear trend analysis
    correlation = corr(x_values, y_values);
    
    if correlation > 0.3
        trend_desc = 'increasing';
    elseif correlation < -0.3
        trend_desc = 'decreasing';
    else
        trend_desc = 'stable';
    end
end

function interpretation = interpret_correlation(corr_val, metric_type)
    % Interpret correlation in context of flatness metrics
    
    if abs(corr_val) < 0.2
        interpretation = '(weak effect)';
    elseif abs(corr_val) < 0.5
        interpretation = '(moderate effect)';
    else
        interpretation = '(strong effect)';
    end
    
    % Add context-specific interpretation
    if strcmp(metric_type, 'flatness') || strcmp(metric_type, 'sharpness')
        if corr_val < -0.3
            interpretation = [interpretation, ' → noise improves flatness'];
        elseif corr_val > 0.3
            interpretation = [interpretation, ' → noise reduces flatness'];
        end
    end
end

% Utility functions (reuse existing ones from your code)
function loss = evaluate_model_loss(params, X_test)
    % Evaluate model loss - enhanced version
    leaky_relu = @(x) max(0.01 * x, x);
    
    total_loss = 0;
    num_samples = size(X_test, 1);
    
    for i = 1:num_samples
        X_sample = X_test(i, :);
        
        % Forward pass with leaky ReLU
        H1_enc = leaky_relu(X_sample * params.We1 + params.be1);
        Z = leaky_relu(H1_enc * params.We_latent + params.be_latent);
        H1_dec = leaky_relu(Z * params.Wd1 + params.bd1);
        X_recon = H1_dec * params.Wd_output + params.bd_output;
        
        % Reconstruction loss
        error = X_recon - X_sample;
        sample_loss = 0.5 * sum(error.^2);
        total_loss = total_loss + sample_loss;
    end
    
    loss = total_loss / num_samples;
end

function perturbed_params = generate_random_perturbation(params, scale)
    perturbed_params = params;
    param_names = fieldnames(params);
    
    for i = 1:length(param_names)
        name = param_names{i};
        noise = scale * randn(size(params.(name)));
        perturbed_params.(name) = params.(name) + noise;
    end
end

function gradients = compute_finite_diff_gradients(params, X_test)
    % Simple finite difference gradients
    h = 1e-5;
    base_loss = evaluate_model_loss(params, X_test);
    
    gradients = struct();
    param_names = fieldnames(params);
    
    for i = 1:length(param_names)
        name = param_names{i};
        param_grad = zeros(size(params.(name)));
        
        % Sample a few elements for efficiency
        [rows, cols] = size(params.(name));
        sample_size = min(100, rows * cols);
        linear_indices = randperm(rows * cols, sample_size);
        
        for idx = linear_indices
            perturbed_params = params;
            perturbed_params.(name)(idx) = params.(name)(idx) + h;
            perturbed_loss = evaluate_model_loss(perturbed_params, X_test);
            param_grad(idx) = (perturbed_loss - base_loss) / h;
        end
        
        gradients.(name) = param_grad;
    end
end

function norm_val = compute_gradient_norm(gradients)
    norm_val = 0;
    param_names = fieldnames(gradients);
    
    for i = 1:length(param_names)
        name = param_names{i};
        norm_val = norm_val + sum(gradients.(name)(:).^2);
    end
    
    norm_val = sqrt(norm_val);
end

function sampled_params = sample_gaussian_params(params, sigma)
    sampled_params = params;
    param_names = fieldnames(params);
    
    for i = 1:length(param_names)
        name = param_names{i};
        noise = sigma * randn(size(params.(name)));
        sampled_params.(name) = params.(name) + noise;
    end
end

function loss = evaluate_loss(params, X_test)
    % Simple loss evaluation for baseline
    relu = @(x) max(0, x);
    
    total_loss = 0;
    num_samples = min(50, size(X_test, 1));
    
    for i = 1:num_samples
        X_sample = X_test(i, :);
        
        % Forward pass
        H1_enc = X_sample * params.We1 + params.be1;
        H1_enc_act = relu(H1_enc);
        Z = H1_enc_act * params.We_latent + params.be_latent;
        Z_act = relu(Z);
        H1_dec = Z_act * params.Wd1 + params.bd1;
        H1_dec_act = relu(H1_dec);
        X_recon = H1_dec_act * params.Wd_output + params.bd_output;
        
        error = X_recon - X_sample;
        sample_loss = 0.5 * sum(error.^2);
        total_loss = total_loss + sample_loss;
    end
    
    loss = total_loss / num_samples;
end

fprintf('\nEnhanced analysis complete. Check the flatness metrics to see how noise affects minima!\n');