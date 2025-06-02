% Author: Kunxin Wu
% Date: 2nd June 2025
%%
clear; clc;

load("./training/weights_log_tensor_leaky.mat");
load("./training/preprocessed_full_data.mat")
addpath(fullfile('.', 'training')); % like this the functions from training become usable
params = weights_log(3).epoch(end);

num_samples = size(X, 1);
num_samples_test = ceil(num_samples * 0.5);

X_test = X(end - num_samples_test + 1:end, :);
X_train = X(1:end - num_samples_test, :);

batch_size = 300;
num_batches = ceil(num_samples_test/batch_size);

fields = fieldnames(params);
% Setup Original Parameters
for i = 1:8
    original_params.(fields{i}) = params.(fields{i});
end
% Training Configuration
config = struct(...
    'learning_rate', 0.0002, ...
    'num_epochs', 10, ...
    'adam_beta1', 0.9, ...
    'adam_beta2', 0.999, ...
    'adam_epsilon', 1e-8);

alpha_leaky = 0.1;

input_size = size(X_train, 2);
hidden_size = 200;
latent_size = 200;
[~, optim, relu, leaky_relu, relu_deriv, leaky_relu_deriv] = setup_network(input_size, hidden_size, latent_size, alpha_leaky); 
batch_descend = 1;
regularization_lambda = 0;

% Noise Analysis Setup
Noise_Levels = [0.001, 0.01, 0.1];
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];

% Initialize Results Storage
loss_histories = {};
final_params_all = {};

% Get baseline performance
original_loss = compute_reconstruction_mse(original_params, X_train, relu);
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
        lastStr = sprintf('\n--- Testing %s with noise %.3f ---\n', current_layer, current_noise);
        fprintf(lastStr);
        % Add noise to specific layers
        noisy_params = add_noise_to_layer(original_params, current_layer, current_noise);
        
        % Evaluate initial damage
        initial_loss = compute_reconstruction_mse(noisy_params, X_train, relu);
        lastStr = sprintf('Initial loss: %.6f\n', initial_loss);
        fprintf(lastStr);

        % Train to recover
        %[final_params, loss_history] = train_adam(noisy_params, X_train, config, relu, relu_derivative);

        [loss_history_after_each_update, loss_history_per_epoch, final_params, final_loss] = train_autoencoder(X_train, noisy_params, optim, relu, relu_deriv,...
        'adam', config.learning_rate, config.num_epochs, regularization_lambda, batch_descend, batch_size, lastStr);
        % Evaluate recovery
        %final_loss = compute_reconstruction_mse(final_params, X_train, relu);
        recovery_pct = ((initial_loss - final_loss) / initial_loss) * 100;
        
        fprintf('Final loss: %.6f\n', final_loss);
        fprintf('Recovery: %.2f%%\n', recovery_pct);
        
        % Store results
        loss_histories{end+1} = loss_history_per_epoch;
        final_params_all{end+1} = final_params;
        results = [results; {current_layer, current_noise, initial_loss, final_loss}];
    end
end

%% === Results Analysis ===
fprintf('\n=== ANALYSIS COMPLETE ===\n');
disp(results);

% Calculate recovery metrics
noisy_results = results(results.NoiseLevel > 0, :);
noisy_results.Recovery = (noisy_results.InitialLoss - noisy_results.FinalLoss) ./ noisy_results.InitialLoss * 100;
noisy_results.ComparedToOriginal = (noisy_results.FinalLoss - original_loss) / original_loss * 100;

fprintf('\n=== RECOVERY METRICS ===\n');
disp(noisy_results);

% Visualization
create_recovery_plots(noisy_results, loss_histories, Layer_Names, Noise_Levels);

% Save Results
save('noise_recovery_results.mat', 'results', 'loss_histories', 'original_params', 'final_params_all');
fprintf('\nResults saved to noise_recovery_results.mat\n');
 

%% SUPPORTING FUNCTIONS

function noisy_params = add_noise_to_layer(params, layer_name, noise_level)
    noisy_params = params;
    rng(123);  % set fixed seed
    W = params.(layer_name);
    noisy_W = W + noise_level * randn(size(W));  % repeatable
    noisy_params.(layer_name) = noisy_W;
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
    escaped_names = strrep(Layer_Names, '_', '\_');
    legend(escaped_names, 'Location', 'best', 'FontSize', 11);    grid on;
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
                    legend_entries{end+1} = sprintf('%s (%.3f)', strrep(Layer_Names(i), '_', '\_'), Noise_Levels(j));
                end
            end
            plot_idx = plot_idx + 1;
        end
    end
    
    xlabel('Epoch', 'FontSize', 12);
    ylabel('Loss', 'FontSize', 12);
    title('Training Convergence Curves', 'FontSize', 14);
    legend(legend_entries, 'Location', 'best', 'FontSize', 9);
    xlim([1 inf]);
    grid on;
    set(gca, 'YScale', 'log', 'FontSize', 11);
    
    sgtitle('Noise Resilience Analysis - Clean Implementation', 'FontSize', 16);
end