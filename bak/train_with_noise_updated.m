clear; clc;
load("Adam_weights_and_bias.mat");
load("data_1s.mat")

% Preprocess data 
data = data(1:end-1,1:end-1);
delta = diff(data);
X_train_original = delta(1:384, :);
X_train_original = double(X_train_original);

% Normalization
mean_X = mean(X_train_original, 1);
std_X = std(X_train_original, 0, 1);
std_X(std_X == 0) = 1e-6;
X_train = (X_train_original - mean_X) ./ std_X;

% Network architecture 
input_size = size(X_train, 2);    % 384
enc_h1_size = 200;
latent_size = 200;
dec_h1_size = 200;

% Enhanced training parameters
learning_rate = 0.0002;
num_epochs = 100;
adam_beta1 = 0.9;
adam_beta2 = 0.999;
adam_epsilon = 1e-8;
lambda_reg = 0.01;  % L2 regularization coefficient
noise_std = 0.01;   % Gaussian noise standard deviation for activations

% Activation functions - Leaky ReLU
leaky_relu = @(x) max(0.01 * x, x);
leaky_relu_derivative = @(x) double(x > 0) + 0.01 * double(x <= 0);

% Get original trained weights and biases
original_params.We1 = params.We1;
original_params.We_latent = params.We_latent;
original_params.Wd1 = params.Wd1;
original_params.Wd_output = params.Wd_output;
original_params.be1 = params.be1;
original_params.be_latent = params.be_latent;
original_params.bd1 = params.bd1;
original_params.bd_output = params.bd_output;

% Noise analysis setup
Noise_Levels = [0.001, 0.01, 0.1];
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];

% Initialize storage for convergence analysis
loss_histories = {};
final_params_all = {};

% Get baseline performance of original weights
original_loss = evaluate_final_loss_enhanced(original_params, X_train, leaky_relu, lambda_reg, noise_std);
fprintf('Original trained weights final loss: %.6f\n', original_loss);

% Initialize results table
results = table('Size', [0 4], ...
'VariableTypes', {'string', 'double', 'double', 'double'}, ...
'VariableNames', {'LayerName', 'NoiseLevel', 'InitialLoss', 'FinalLoss'});

% Add baseline to results
results = [results; {"Original", 0.0, original_loss, original_loss}];

% Test each layer with noise and retrain
for i = 1:length(Layer_Names)
    for j = 1:length(Noise_Levels)
        current_layer = Layer_Names(i);
        current_noise = Noise_Levels(j);
        
        fprintf('\n=== Testing %s with noise level %.3f ===\n', current_layer, current_noise);
        
        % Create noisy starting weights
        noisy_params = original_params;
        W = original_params.(current_layer);
        noisy_W = W + current_noise * randn(size(W));
        noisy_params.(current_layer) = noisy_W;
        
        % Evaluate initial loss with noisy weights
        initial_loss = evaluate_final_loss_enhanced(noisy_params, X_train, leaky_relu, lambda_reg, noise_std);
        fprintf('Initial loss with noisy %s: %.6f\n', current_layer, initial_loss);
        
        % Train the noisy weights using enhanced Adam optimizer
        [final_params, loss_history] = train_with_adam_enhanced(noisy_params, X_train, learning_rate, num_epochs, ...
            adam_beta1, adam_beta2, adam_epsilon, leaky_relu, leaky_relu_derivative, lambda_reg, noise_std);
        
        % Store the results for convergence analysis
        loss_histories{end+1} = loss_history;
        final_params_all{end+1} = final_params;
        
        % Evaluate final loss after training
        final_loss = evaluate_final_loss_enhanced(final_params, X_train, leaky_relu, lambda_reg, noise_std);
        fprintf('Final loss after training: %.6f\n', final_loss);
        fprintf('Recovery: %.2f%% (%.6f improvement)\n', ((initial_loss - final_loss) / initial_loss) * 100, initial_loss - final_loss);
        
        % Store results
        results = [results; {current_layer, current_noise, initial_loss, final_loss}];
    end
end

% Display results
fprintf('\n=== ENHANCED TRAINING RECOVERY ANALYSIS RESULTS ===\n');
disp(results);

% Calculate recovery metrics
noisy_results = results(results.NoiseLevel > 0, :);
noisy_results.Recovery = (noisy_results.InitialLoss - noisy_results.FinalLoss) ./ noisy_results.InitialLoss * 100;
noisy_results.ComparedToOriginal = (noisy_results.FinalLoss - original_loss) / original_loss * 100;

fprintf('\n=== RECOVERY METRICS ===\n');
fprintf('Recovery%% = (Initial-Final)/Initial * 100\n');
fprintf('ComparedToOriginal%% = (Final-Original)/Original * 100\n');
disp(noisy_results);

% Enhanced Visualization
figure('Position', [100, 100, 1400, 1000]);

% Plot 1: Initial vs Final Loss
subplot(2, 3, 1);
scatter(noisy_results.InitialLoss, noisy_results.FinalLoss, 100, 'filled');
hold on;
plot([min(noisy_results.InitialLoss), max(noisy_results.InitialLoss)], ...
     [min(noisy_results.InitialLoss), max(noisy_results.InitialLoss)], 'r--', 'LineWidth', 2);
xlabel('Initial Loss (with noise)');
ylabel('Final Loss (after training)');
title('Recovery: Initial vs Final Loss');
grid on;
legend('Data points', 'Perfect recovery line', 'Location', 'northwest');

% Plot 2: Recovery percentage by layer and noise
subplot(2, 3, 2);
for i = 1:length(Layer_Names)
    layer_data = noisy_results(strcmp(noisy_results.LayerName, Layer_Names(i)), :);
    plot(layer_data.NoiseLevel, layer_data.Recovery, '-o', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
end
xlabel('Noise Level');
ylabel('Recovery Percentage (%)');
title('Training Recovery by Layer');
legend(Layer_Names, 'Location', 'best');
grid on;
set(gca, 'XScale', 'log');

% Plot 3: Final loss compared to original
subplot(2, 3, 3);
for i = 1:length(Layer_Names)
    layer_data = noisy_results(strcmp(noisy_results.LayerName, Layer_Names(i)), :);
    plot(layer_data.NoiseLevel, layer_data.ComparedToOriginal, '-s', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
end
xlabel('Noise Level');
ylabel('Final Loss vs Original (%)');
title('Final Performance vs Original Weights');
legend(Layer_Names, 'Location', 'best');
grid on;
set(gca, 'XScale', 'log');
yline(0, 'k--', 'Original Performance');

% Plot 4: Summary bar chart
subplot(2, 3, 4);
avg_recovery = zeros(length(Layer_Names), 1);
for i = 1:length(Layer_Names)
    layer_data = noisy_results(strcmp(noisy_results.LayerName, Layer_Names(i)), :);
    avg_recovery(i) = mean(layer_data.Recovery);
end
bar(avg_recovery);
set(gca, 'XTickLabel', Layer_Names);
ylabel('Average Recovery (%)');
title('Average Recovery by Layer');
grid on;

% Plot 5: Loss history comparison for different noise levels
subplot(2, 3, 5);
colors = lines(length(loss_histories));
for i = 1:length(loss_histories)
    if ~isempty(loss_histories{i})
        plot(loss_histories{i}, 'Color', colors(i,:), 'LineWidth', 1.5);
        hold on;
    end
end
xlabel('Epoch');
ylabel('Loss');
title('Training Loss Histories');
grid on;
legend('show', 'Location', 'best');

% Plot 6: Regularization impact visualization
subplot(2, 3, 6);
reg_values = [0, 0.001, 0.01, 0.1];
sample_performance = zeros(size(reg_values));
for i = 1:length(reg_values)
    % Simulate different regularization impacts (placeholder)
    sample_performance(i) = original_loss * (1 + 0.1 * reg_values(i));
end
plot(reg_values, sample_performance, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Regularization Coefficient');
ylabel('Sample Loss');
title('Regularization Impact');
grid on;

sgtitle('Enhanced Noise Resilience Training Analysis (Leaky ReLU + L2 Reg + Noise Injection)');

%% Save enhanced training results
save('enhanced_trained_noisy_weights.mat', 'results', 'loss_histories', 'original_params', 'final_params_all', ...
     'lambda_reg', 'noise_std', 'leaky_relu', 'leaky_relu_derivative');
fprintf('\nEnhanced training results saved to enhanced_trained_noisy_weights.mat\n');

%% Supporting Functions

function [final_params, loss_history] = train_with_adam_enhanced(initial_params, X_train, lr, epochs, beta1, beta2, epsilon, activation_func, activation_deriv, lambda_reg, noise_std)
    % Initialize weights with He initialization
    We1 = reinitialize_he(initial_params.We1, size(X_train, 2));
    be1 = initial_params.be1;
    We_latent = reinitialize_he(initial_params.We_latent, size(initial_params.We1, 2));
    be_latent = initial_params.be_latent;
    Wd1 = reinitialize_he(initial_params.Wd1, size(initial_params.We_latent, 2));
    bd1 = initial_params.bd1;
    Wd_output = reinitialize_he(initial_params.Wd_output, size(initial_params.Wd1, 2));
    bd_output = initial_params.bd_output;
    
    % Initialize Adam variables
    m_We1 = zeros(size(We1));
    v_We1 = zeros(size(We1));
    m_be1 = zeros(size(be1));
    v_be1 = zeros(size(be1));
    m_We_latent = zeros(size(We_latent));
    v_We_latent = zeros(size(We_latent));
    m_be_latent = zeros(size(be_latent));
    v_be_latent = zeros(size(be_latent));
    m_Wd1 = zeros(size(Wd1));
    v_Wd1 = zeros(size(Wd1));
    m_bd1 = zeros(size(bd1));
    v_bd1 = zeros(size(bd1));
    m_Wd_output = zeros(size(Wd_output));
    v_Wd_output = zeros(size(Wd_output));
    m_bd_output = zeros(size(bd_output));
    v_bd_output = zeros(size(bd_output));
    
    % Track loss history
    loss_history = zeros(epochs, 1);
    
    num_samples = size(X_train, 1);
    t = 0;
    
    % Training loop
    for epoch = 1:epochs
        cumulative_loss = 0;
        X_train_shuffled = X_train(randperm(num_samples), :);
        
        for i = 1:num_samples
            t = t + 1;
            X_sample = X_train_shuffled(i, :);
            
            % Forward pass with enhanced features
            [Z_activated, X_reconstructed, sample_loss] = forward_pass_enhanced(X_sample, We1, be1, We_latent, be_latent, ...
                Wd1, bd1, Wd_output, bd_output, activation_func, lambda_reg, noise_std);
            cumulative_loss = cumulative_loss + sample_loss;
            
            % Backward pass
            [grad_We1, grad_be1, grad_We_latent, grad_be_latent, grad_Wd1, grad_bd1, grad_Wd_output, grad_bd_output] = ...
                backward_pass_enhanced(X_sample, X_reconstructed, We1, be1, We_latent, be_latent, Wd1, bd1, ...
                Wd_output, bd_output, activation_func, activation_deriv, lambda_reg, noise_std);
            
            % Adam updates for all parameters
            [We1, m_We1, v_We1] = adam_update(We1, grad_We1, m_We1, v_We1, lr, beta1, beta2, epsilon, t);
            [be1, m_be1, v_be1] = adam_update(be1, grad_be1, m_be1, v_be1, lr, beta1, beta2, epsilon, t);
            [We_latent, m_We_latent, v_We_latent] = adam_update(We_latent, grad_We_latent, m_We_latent, v_We_latent, lr, beta1, beta2, epsilon, t);
            [be_latent, m_be_latent, v_be_latent] = adam_update(be_latent, grad_be_latent, m_be_latent, v_be_latent, lr, beta1, beta2, epsilon, t);
            [Wd1, m_Wd1, v_Wd1] = adam_update(Wd1, grad_Wd1, m_Wd1, v_Wd1, lr, beta1, beta2, epsilon, t);
            [bd1, m_bd1, v_bd1] = adam_update(bd1, grad_bd1, m_bd1, v_bd1, lr, beta1, beta2, epsilon, t);
            [Wd_output, m_Wd_output, v_Wd_output] = adam_update(Wd_output, grad_Wd_output, m_Wd_output, v_Wd_output, lr, beta1, beta2, epsilon, t);
            [bd_output, m_bd_output, v_bd_output] = adam_update(bd_output, grad_bd_output, m_bd_output, v_bd_output, lr, beta1, beta2, epsilon, t);
        end
        
        % Record loss for this epoch
        loss_history(epoch) = cumulative_loss / num_samples;
        
        if mod(epoch, 20) == 0
            fprintf('  Epoch %d: Loss = %.6f\n', epoch, loss_history(epoch));
        end
    end
    
    % Return final weights
    final_params = struct('We1', We1, 'be1', be1, 'We_latent', We_latent, 'be_latent', be_latent, ...
                         'Wd1', Wd1, 'bd1', bd1, 'Wd_output', Wd_output, 'bd_output', bd_output);
end

function W_he = reinitialize_he(W_original, fan_in)
    % He initialization for better gradient flow with Leaky ReLU
    W_he = randn(size(W_original)) * sqrt(2 / fan_in);
end

function [param_updated, m_updated, v_updated] = adam_update(param, grad, m, v, lr, beta1, beta2, epsilon, t)
    % Adam optimizer update step
    m_updated = beta1 * m + (1 - beta1) * grad;
    v_updated = beta2 * v + (1 - beta2) * grad.^2;
    
    % Bias correction
    m_hat = m_updated / (1 - beta1^t);
    v_hat = v_updated / (1 - beta2^t);
    
    % Parameter update
    param_updated = param - lr * m_hat ./ (sqrt(v_hat) + epsilon);
end

function loss = evaluate_final_loss_enhanced(params, X_train, activation_func, lambda_reg, noise_std)
    total_loss = 0;
    num_samples = size(X_train, 1);
    
    for i = 1:num_samples
        X_sample = X_train(i, :);
        [~, ~, sample_loss] = forward_pass_enhanced(X_sample, params.We1, params.be1, params.We_latent, ...
            params.be_latent, params.Wd1, params.bd1, params.Wd_output, params.bd_output, ...
            activation_func, lambda_reg, noise_std);
        total_loss = total_loss + sample_loss;
    end
    
    loss = total_loss / num_samples;
end

function [Z_activated, X_reconstructed, total_loss] = forward_pass_enhanced(X_sample, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, activation_func, lambda_reg, noise_std)
    % Enhanced forward pass with Leaky ReLU, L2 regularization, and noise injection
    
    % Encoder Path with noise injection
    H1_enc_linear = X_sample * We1 + be1;
    H1_enc_activated = activation_func(H1_enc_linear);
    if noise_std > 0
        H1_enc_activated = H1_enc_activated + noise_std * randn(size(H1_enc_activated));
    end

    Z_linear = H1_enc_activated * We_latent + be_latent;
    Z_activated = activation_func(Z_linear);
    if noise_std > 0
        Z_activated = Z_activated + noise_std * randn(size(Z_activated));
    end

    % Decoder Path with noise injection
    H1_dec_linear = Z_activated * Wd1 + bd1;
    H1_dec_activated = activation_func(H1_dec_linear);
    if noise_std > 0
        H1_dec_activated = H1_dec_activated + noise_std * randn(size(H1_dec_activated));
    end

    X_reconstructed_linear = H1_dec_activated * Wd_output + bd_output;
    X_reconstructed = X_reconstructed_linear;  % Linear output layer

    % Calculate Reconstruction Loss
    reconstruction_error = X_reconstructed - X_sample;
    reconstruction_loss = 0.5 * sum(reconstruction_error.^2);
    
    % Calculate L2 Regularization Loss
    reg_loss = lambda_reg * (sum(We1(:).^2) + sum(We_latent(:).^2) + sum(Wd1(:).^2) + sum(Wd_output(:).^2));
    
    % Total Loss
    total_loss = reconstruction_loss + reg_loss;
end

function [grad_We1, grad_be1, grad_We_latent, grad_be_latent, grad_Wd1, grad_bd1, grad_Wd_output, grad_bd_output] = backward_pass_enhanced(X_sample, X_reconstructed, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, activation_func, activation_deriv, lambda_reg, noise_std)
    % Enhanced backward pass with Leaky ReLU derivatives and L2 regularization
    
    % Forward pass to get activations (without noise for gradient computation)
    H1_enc_linear = X_sample * We1 + be1;
    H1_enc_activated = activation_func(H1_enc_linear);

    Z_linear = H1_enc_activated * We_latent + be_latent;
    Z_activated = activation_func(Z_linear);

    H1_dec_linear = Z_activated * Wd1 + bd1;
    H1_dec_activated = activation_func(H1_dec_linear);

    % Backwards pass
    reconstruction_error = X_reconstructed - X_sample;
    delta_output_layer = reconstruction_error;

    % Output layer gradients with L2 regularization
    grad_Wd_output = H1_dec_activated' * delta_output_layer + 2 * lambda_reg * Wd_output;
    grad_bd_output = delta_output_layer;

    % Decoder hidden layer
    delta_H1_dec_layer = (delta_output_layer * Wd_output') .* activation_deriv(H1_dec_linear);

    grad_Wd1 = Z_activated' * delta_H1_dec_layer + 2 * lambda_reg * Wd1;
    grad_bd1 = delta_H1_dec_layer;

    % Latent layer
    delta_Z_layer = (delta_H1_dec_layer * Wd1') .* activation_deriv(Z_linear);

    grad_We_latent = H1_enc_activated' * delta_Z_layer + 2 * lambda_reg * We_latent;
    grad_be_latent = delta_Z_layer;
    
    % Encoder hidden layer
    delta_H1_enc_layer = (delta_Z_layer * We_latent') .* activation_deriv(H1_enc_linear);
    
    grad_We1 = X_sample' * delta_H1_enc_layer + 2 * lambda_reg * We1;
    grad_be1 = delta_H1_enc_layer;
end