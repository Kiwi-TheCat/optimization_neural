% Author: Kunxin Wu
% Date: 2nd June 2025
%%
clear; clc;

% Load data and pre-trained weights
load("Adam_weights_and_bias.mat");
load("data_1s.mat")

%% Data Preprocessing
data = data(1:end-1,1:end-1);
delta = diff(data);
X_train_original = delta(1:384, :);
X_train_original = double(X_train_original);

% Normalization
mean_X = mean(X_train_original, 1);
std_X = std(X_train_original, 0, 1);
std_X(std_X == 0) = 1e-6;
X_train = (X_train_original - mean_X) ./ std_X;

%% Setup Original Parameters
original_params = struct(...
    'We1', params.We1, 'be1', params.be1, ...
    'We_latent', params.We_latent, 'be_latent', params.be_latent, ...
    'Wd1', params.Wd1, 'bd1', params.bd1, ...
    'Wd_output', params.Wd_output, 'bd_output', params.bd_output);

%% Training Configuration
config = struct(...
    'learning_rate', 0.0002, ...
    'num_epochs', 100, ...
    'adam_beta1', 0.9, ...
    'adam_beta2', 0.999, ...
    'adam_epsilon', 1e-8);

% Activation functions
relu = @(x) max(0, x);
relu_derivative = @(a) double(a > 0);

%% Noise Analysis Setup
Noise_Levels = [0.001, 0.01, 0.1];
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];

%% Initialize Results Storage
loss_histories = {};
final_params_all = {};

% Get baseline performance
original_loss = evaluate_loss(original_params, X_train, relu);
fprintf('Original trained weights final loss: %.6f\n', original_loss);

% Initialize results table
results = table('Size', [0 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'LayerName', 'NoiseLevel', 'InitialLoss', 'FinalLoss'});

results = [results; {"Original", 0.0, original_loss, original_loss}];

%% Main Analysis Loop
fprintf('\n=== STARTING NOISE RECOVERY ANALYSIS ===\n');

for i = 1:length(Layer_Names)
    for j = 1:length(Noise_Levels)
        current_layer = Layer_Names(i);
        current_noise = Noise_Levels(j);
        
        fprintf('\n--- Testing %s with noise %.3f ---\n', current_layer, current_noise);
        
        % Add noise to specific layer
        noisy_params = add_noise_to_layer(original_params, current_layer, current_noise);
        
        % Evaluate initial damage
        initial_loss = evaluate_loss(noisy_params, X_train, relu);
        fprintf('Initial loss: %.6f\n', initial_loss);
        
        % Train to recover
        [final_params, loss_history] = train_adam(noisy_params, X_train, config, relu, relu_derivative);
        
        % Evaluate recovery
        final_loss = evaluate_loss(final_params, X_train, relu);
        recovery_pct = ((initial_loss - final_loss) / initial_loss) * 100;
        
        fprintf('Final loss: %.6f\n', final_loss);
        fprintf('Recovery: %.2f%%\n', recovery_pct);
        
        % Store results
        loss_histories{end+1} = loss_history;
        final_params_all{end+1} = final_params;
        results = [results; {current_layer, current_noise, initial_loss, final_loss}];
    end
end

%% Results Analysis
fprintf('\n=== ANALYSIS COMPLETE ===\n');
disp(results);

% Calculate recovery metrics
noisy_results = results(results.NoiseLevel > 0, :);
noisy_results.Recovery = (noisy_results.InitialLoss - noisy_results.FinalLoss) ./ noisy_results.InitialLoss * 100;
noisy_results.ComparedToOriginal = (noisy_results.FinalLoss - original_loss) / original_loss * 100;

fprintf('\n=== RECOVERY METRICS ===\n');
disp(noisy_results);

%% Visualization
create_recovery_plots(noisy_results, loss_histories, Layer_Names, Noise_Levels);

%% Save Results
save('noise_recovery_results.mat', 'results', 'loss_histories', 'original_params', 'final_params_all');
fprintf('\nResults saved to noise_recovery_results.mat\n');
 

%% SUPPORTING FUNCTIONS

function noisy_params = add_noise_to_layer(params, layer_name, noise_level)
    noisy_params = params;
    W = params.(layer_name);
    noisy_params.(layer_name) = W + noise_level * randn(size(W));
end

function loss = evaluate_loss(params, X_train, relu_func)
    total_loss = 0;
    num_samples = size(X_train, 1);
    
    for i = 1:num_samples
        x = X_train(i, :);
        x_reconstructed = forward_pass(x, params, relu_func);
        reconstruction_error = x_reconstructed - x;
        total_loss = total_loss + 0.5 * sum(reconstruction_error.^2);
    end
    
    loss = total_loss / num_samples;
end

function x_reconstructed = forward_pass(x, params, relu)
    % Encoder
    h1 = relu(x * params.We1 + params.be1);
    z = relu(h1 * params.We_latent + params.be_latent);
    
    % Decoder
    h2 = relu(z * params.Wd1 + params.bd1);
    x_reconstructed = h2 * params.Wd_output + params.bd_output;
end

function [final_params, loss_history] = train_adam(initial_params, X_train, config, relu, relu_deriv)
    % Initialize parameters
    params = initial_params;
    num_samples = size(X_train, 1);
    loss_history = zeros(config.num_epochs, 1);
    
    % Initialize Adam optimizer state
    fields = fieldnames(params);
    m = struct(); v = struct();
    for i = 1:length(fields)
        m.(fields{i}) = zeros(size(params.(fields{i})));
        v.(fields{i}) = zeros(size(params.(fields{i})));
    end
    
    t = 0; % Global time step
    
    % Training loop
    for epoch = 1:config.num_epochs
        X_shuffled = X_train(randperm(num_samples), :);
        total_loss = 0;
        
        for i = 1:num_samples
            t = t + 1;
            x = X_shuffled(i, :);
            
            % Forward and backward pass
            [loss, grads] = compute_gradients(x, params, relu, relu_deriv);
            total_loss = total_loss + loss;
            
            % Adam parameter updates
            for j = 1:length(fields)
                field = fields{j};
                g = grads.(field);
                
                % Update biased first and second moment estimates
                m.(field) = config.adam_beta1 * m.(field) + (1 - config.adam_beta1) * g;
                v.(field) = config.adam_beta2 * v.(field) + (1 - config.adam_beta2) * g.^2;
                
                % Bias correction
                m_hat = m.(field) / (1 - config.adam_beta1^t);
                v_hat = v.(field) / (1 - config.adam_beta2^t);
                
                % Parameter update
                params.(field) = params.(field) - config.learning_rate * m_hat ./ (sqrt(v_hat) + config.adam_epsilon);
            end
        end
        
        loss_history(epoch) = total_loss / num_samples;
        
        if mod(epoch, 20) == 0
            fprintf('  Epoch %d: Loss = %.6f\n', epoch, loss_history(epoch));
        end
    end
    
    final_params = params;
end

function [loss, grads] = compute_gradients(x, params, relu, relu_deriv)
    % Forward pass with intermediate values stored
    h1_linear = x * params.We1 + params.be1;
    h1 = relu(h1_linear);
    
    z_linear = h1 * params.We_latent + params.be_latent;
    z = relu(z_linear);
    
    h2_linear = z * params.Wd1 + params.bd1;
    h2 = relu(h2_linear);
    
    x_reconstructed = h2 * params.Wd_output + params.bd_output;
    
    % Loss
    reconstruction_error = x_reconstructed - x;
    loss = 0.5 * sum(reconstruction_error.^2);
    
    % Backward pass
    dL_dx_recon = reconstruction_error;
    
    % Output layer
    grads.Wd_output = h2' * dL_dx_recon;
    grads.bd_output = dL_dx_recon;
    
    % Hidden layer 2
    dL_dh2 = (dL_dx_recon * params.Wd_output') .* relu_deriv(h2);
    grads.Wd1 = z' * dL_dh2;
    grads.bd1 = dL_dh2;
    
    % Latent layer
    dL_dz = (dL_dh2 * params.Wd1') .* relu_deriv(z);
    grads.We_latent = h1' * dL_dz;
    grads.be_latent = dL_dz;
    
    % Input layer
    dL_dh1 = (dL_dz * params.We_latent') .* relu_deriv(h1);
    grads.We1 = x' * dL_dh1;
    grads.be1 = dL_dh1;
end

function create_recovery_plots(noisy_results, loss_histories, Layer_Names, Noise_Levels)
    figure('Position', [100, 100, 1200, 500]);
    
    % Plot 1: Recovery by Layer
    subplot(1, 2, 1);
    colors = lines(length(Layer_Names));
    for i = 1:length(Layer_Names)
        layer_data = noisy_results(strcmp(noisy_results.LayerName, Layer_Names(i)), :);
        plot(layer_data.NoiseLevel, layer_data.Recovery, '-o', ...
             'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(i,:));
        hold on;
    end
    xlabel('Noise Level', 'FontSize', 12);
    ylabel('Recovery Percentage (%)', 'FontSize', 12);
    title('Training Recovery by Layer', 'FontSize', 14);
    legend(Layer_Names, 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'XScale', 'log', 'FontSize', 11);
    
    % Plot 2: Loss Convergence
    subplot(1, 2, 2);
    legend_entries = {};
    plot_idx = 1;
    
    for i = 1:length(Layer_Names)
        for j = 1:length(Noise_Levels)
            if plot_idx <= length(loss_histories)
                loss_curve = loss_histories{plot_idx};
                if ~isempty(loss_curve)
                    line_styles = {'-', '--', ':'};
                    plot(1:length(loss_curve), loss_curve, line_styles{j}, ...
                         'LineWidth', 1.8, 'Color', colors(i,:));
                    hold on;
                    legend_entries{end+1} = sprintf('%s (%.3f)', Layer_Names(i), Noise_Levels(j));
                end
            end
            plot_idx = plot_idx + 1;
        end
    end
    
    xlabel('Epoch', 'FontSize', 12);
    ylabel('Loss', 'FontSize', 12);
    title('Training Convergence Curves', 'FontSize', 14);
    legend(legend_entries, 'Location', 'best', 'FontSize', 9);
    grid on;
    set(gca, 'YScale', 'log', 'FontSize', 11);
    
    sgtitle('Noise Resilience Analysis - Clean Implementation', 'FontSize', 16);
end