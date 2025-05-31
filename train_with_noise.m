%% Step 1: Load the best weight matrix of Adam method
clear; clc;
load("Adam_weights_and_bias.mat");
load("data_1s.mat")

% Preprocess data
data = data(1:end-1,1:end-1);
delta = diff(data);
X_train_original = delta(1:384, :);
X_train_original = double(X_train_original);

% Access original weights from a specific epoch
original_weights.We1        = params.We1;
original_weights.We_latent  = params.We_latent;
original_weights.Wd1        = params.Wd1;
original_weights.Wd_output  = params.Wd_output;
% Access original biases
original_bias.be1           = params.be1;
original_bias.be_latent     = params.be_latent;
original_bias.bd1           = params.bd1;
original_bias.bd_output     = params.bd_output;

% Normalization parameters (same as in your main script)
mean_X = mean(X_train_original, 1);
std_X = std(X_train_original, 0, 1);
std_X(std_X == 0) = 1e-6;

% ReLU function
relu = @(x) max(0, x);

% Noise analysis setup
Noise_Levels = [0.001, 0.01, 0.1];
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];

% Initialize results table
results = table('Size', [0 3], ...
'VariableTypes', {'string', 'double', 'double'}, ...
'VariableNames', {'LayerName', 'Noise', 'Loss'});

% Test original model first (baseline)
original_loss = evaluate_model_loss(original_weights, original_bias, X_train_original, mean_X, std_X, relu);
fprintf('Original model loss (no noise): %.6f\n', original_loss);
results = [results; {"Original", 0.0, original_loss}];

% Loop through each layer and noise level
for i = 1:length(Layer_Names)
    for j = 1:length(Noise_Levels)
        current_layer = Layer_Names(i);
        current_noise = Noise_Levels(j);
        
        % Create copies of original weights and biases
        noisy_weights = original_weights;
        noisy_bias = original_bias;
        
        % Add noise to the current layer only
        W = original_weights.(current_layer);
        noisy_W = W + current_noise * randn(size(W));
        noisy_weights.(current_layer) = noisy_W;
        
        % Evaluate loss with noisy weights
        current_loss = evaluate_model_loss(noisy_weights, noisy_bias, X_train_original, mean_X, std_X, relu);
        
        % Store results
        results = [results; {current_layer, current_noise, current_loss}];
        
        fprintf('Layer: %s, Noise: %.3f, Loss: %.6f\n', current_layer, current_noise, current_loss);
    end
end

% Display results
fprintf('\n=== NOISE ANALYSIS RESULTS ===\n');
disp(results);

% Simple visualization
figure('Position', [100, 100, 800, 600]);
noise_data = results(results.Noise > 0, :);

% Plot loss vs noise for each layer
hold on;
colors = ['b', 'r', 'g', 'm'];
for i = 1:length(Layer_Names)
    layer_data = noise_data(strcmp(noise_data.LayerName, Layer_Names(i)), :);
    plot(layer_data.Noise, layer_data.Loss, ['-o' colors(i)], 'LineWidth', 2, 'MarkerSize', 8);
end
hold off;

xlabel('Noise Level');
ylabel('Reconstruction Loss');
title('Noise Sensitivity Analysis');
legend(Layer_Names, 'Location', 'northwest');
grid on;
set(gca, 'XScale', 'log'); % Log scale for noise levels

% Supporting function
function loss = evaluate_model_loss(weights, biases, test_data, mean_X, std_X, relu_func)
    % Normalize test data
    test_data_norm = (test_data - mean_X) ./ std_X;
    
    total_loss = 0;
    num_samples = size(test_data_norm, 1);
    
    % Use only a subset for faster evaluation
    num_test_samples = min(50, num_samples);
    
    for i = 1:num_test_samples
        X_sample = test_data_norm(i, :);
        
        % Forward pass through encoder
        H1_enc = X_sample * weights.We1 + biases.be1;
        H1_enc_act = relu_func(H1_enc);
        
        Z = H1_enc_act * weights.We_latent + biases.be_latent;
        Z_act = relu_func(Z);
        
        % Forward pass through decoder
        H1_dec = Z_act * weights.Wd1 + biases.bd1;
        H1_dec_act = relu_func(H1_dec);
        
        X_recon = H1_dec_act * weights.Wd_output + biases.bd_output;
        
        % Calculate reconstruction loss
        error = X_recon - X_sample;
        sample_loss = 0.5 * sum(error.^2);
        total_loss = total_loss + sample_loss;
    end
    
    loss = total_loss / num_test_samples;
end

%% training

% Training parameters (same as your main script)
learning_rate = 0.0002;
num_epochs = 100; % Reduced for testing
adam_beta1 = 0.9;
adam_beta2 = 0.999;
adam_epsilon = 1e-8;
relu = @(x) max(0, x);
relu_derivative = @(a) double(a > 0);

% Noise analysis setup
Noise_Levels = [0.001, 0.01, 0.1];
Layer_Names = ["We1", "We_latent", "Wd1", "Wd_output"];

% Get baseline performance of original weights
original_loss = evaluate_final_loss(original_weights, X_train_original, relu);
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
        noisy_weights = original_weights;
        W = original_weights.(current_layer);
        noisy_W = W + current_noise * randn(size(W));
        noisy_weights.(current_layer) = noisy_W;
        
        % Evaluate initial loss with noisy weights
        initial_loss = evaluate_final_loss(noisy_weights, X_train, relu);
        fprintf('Initial loss with noisy %s: %.6f\n', current_layer, initial_loss);
        
        % Train the noisy weights using Adam optimizer
        final_weights = train_with_adam(noisy_weights, X_train, learning_rate, num_epochs, adam_beta1, adam_beta2, adam_epsilon, relu, relu_derivative);
        
        % Evaluate final loss after training
        final_loss = evaluate_final_loss(final_weights, X_train, relu);
        fprintf('Final loss after training: %.6f\n', final_loss);
        fprintf('Recovery: %.2f%% (%.6f improvement)\n', ((initial_loss - final_loss) / initial_loss) * 100, initial_loss - final_loss);
        
        % Store results
        results = [results; {current_layer, current_noise, initial_loss, final_loss}];
    end
end

% Display results
fprintf('\n=== TRAINING RECOVERY ANALYSIS RESULTS ===\n');
disp(results);

% Calculate recovery metrics
noisy_results = results(results.NoiseLevel > 0, :);
noisy_results.Recovery = (noisy_results.InitialLoss - noisy_results.FinalLoss) ./ noisy_results.InitialLoss * 100;
noisy_results.ComparedToOriginal = (noisy_results.FinalLoss - original_loss) / original_loss * 100;

fprintf('\n=== RECOVERY METRICS ===\n');
fprintf('Recovery%% = (Initial-Final)/Initial * 100\n');
fprintf('ComparedToOriginal%% = (Final-Original)/Original * 100\n');
disp(noisy_results);

% Visualization
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Initial vs Final Loss
subplot(2, 2, 1);
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
subplot(2, 2, 2);
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
subplot(2, 2, 3);
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
subplot(2, 2, 4);
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

sgtitle('Noise Resilience Training Analysis');

%% Supporting Functions

function final_weights = train_with_adam(initial_weights, X_train, lr, epochs, beta1, beta2, epsilon, relu, relu_deriv)
    % Initialize weights
    We1 = initial_weights.We1;
    be1 = initial_weights.be1;
    We_latent = initial_weights.We_latent;
    be_latent = initial_weights.be_latent;
    Wd1 = initial_weights.Wd1;
    bd1 = initial_weights.bd1;
    Wd_output = initial_weights.Wd_output;
    bd_output = initial_weights.bd_output;
    
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
    
    num_samples = size(X_train, 1);
    t = 0;
    
    % Training loop
    for epoch = 1:epochs
        X_train_shuffled = X_train(randperm(num_samples), :);
        
        for i = 1:num_samples
            t = t + 1;
            X_sample = X_train_shuffled(i, :);
            
            % Forward pass
            [Z_activated, X_reconstructed, ~] = forward_pass(X_sample, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu);
            
            % Backward pass
            [grad_We1, grad_be1, grad_We_latent, grad_be_latent, grad_Wd1, grad_bd1, grad_Wd_output, grad_bd_output] = ...
                backward_pass(X_sample, X_reconstructed, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu, relu_deriv);
            
            % Adam updates
            % We1
            m_We1 = beta1 * m_We1 + (1 - beta1) * grad_We1;
            v_We1 = beta2 * v_We1 + (1 - beta2) * grad_We1.^2;
            m_hat_We1 = m_We1 / (1 - beta1^t);
            v_hat_We1 = v_We1 / (1 - beta2^t);
            We1 = We1 - lr * m_hat_We1 ./ (sqrt(v_hat_We1) + epsilon);
            
            % be1
            m_be1 = beta1 * m_be1 + (1 - beta1) * grad_be1;
            v_be1 = beta2 * v_be1 + (1 - beta2) * grad_be1.^2;
            m_hat_be1 = m_be1 / (1 - beta1^t);
            v_hat_be1 = v_be1 / (1 - beta2^t);
            be1 = be1 - lr * m_hat_be1 ./ (sqrt(v_hat_be1) + epsilon);
            
            % We_latent
            m_We_latent = beta1 * m_We_latent + (1 - beta1) * grad_We_latent;
            v_We_latent = beta2 * v_We_latent + (1 - beta2) * grad_We_latent.^2;
            m_hat_We_latent = m_We_latent / (1 - beta1^t);
            v_hat_We_latent = v_We_latent / (1 - beta2^t);
            We_latent = We_latent - lr * m_hat_We_latent ./ (sqrt(v_hat_We_latent) + epsilon);
            
            % be_latent
            m_be_latent = beta1 * m_be_latent + (1 - beta1) * grad_be_latent;
            v_be_latent = beta2 * v_be_latent + (1 - beta2) * grad_be_latent.^2;
            m_hat_be_latent = m_be_latent / (1 - beta1^t);
            v_hat_be_latent = v_be_latent / (1 - beta2^t);
            be_latent = be_latent - lr * m_hat_be_latent ./ (sqrt(v_hat_be_latent) + epsilon);
            
            % Wd1
            m_Wd1 = beta1 * m_Wd1 + (1 - beta1) * grad_Wd1;
            v_Wd1 = beta2 * v_Wd1 + (1 - beta2) * grad_Wd1.^2;
            m_hat_Wd1 = m_Wd1 / (1 - beta1^t);
            v_hat_Wd1 = v_Wd1 / (1 - beta2^t);
            Wd1 = Wd1 - lr * m_hat_Wd1 ./ (sqrt(v_hat_Wd1) + epsilon);
            
            % bd1
            m_bd1 = beta1 * m_bd1 + (1 - beta1) * grad_bd1;
            v_bd1 = beta2 * v_bd1 + (1 - beta2) * grad_bd1.^2;
            m_hat_bd1 = m_bd1 / (1 - beta1^t);
            v_hat_bd1 = v_bd1 / (1 - beta2^t);
            bd1 = bd1 - lr * m_hat_bd1 ./ (sqrt(v_hat_bd1) + epsilon);
            
            % Wd_output
            m_Wd_output = beta1 * m_Wd_output + (1 - beta1) * grad_Wd_output;
            v_Wd_output = beta2 * v_Wd_output + (1 - beta2) * grad_Wd_output.^2;
            m_hat_Wd_output = m_Wd_output / (1 - beta1^t);
            v_hat_Wd_output = v_Wd_output / (1 - beta2^t);
            Wd_output = Wd_output - lr * m_hat_Wd_output ./ (sqrt(v_hat_Wd_output) + epsilon);
            
            % bd_output
            m_bd_output = beta1 * m_bd_output + (1 - beta1) * grad_bd_output;
            v_bd_output = beta2 * v_bd_output + (1 - beta2) * grad_bd_output.^2;
            m_hat_bd_output = m_bd_output / (1 - beta1^t);
            v_hat_bd_output = v_bd_output / (1 - beta2^t);
            bd_output = bd_output - lr * m_hat_bd_output ./ (sqrt(v_hat_bd_output) + epsilon);
        end
        
        if mod(epoch, 20) == 0
            current_loss = evaluate_final_loss(struct('We1', We1, 'be1', be1, 'We_latent', We_latent, 'be_latent', be_latent, 'Wd1', Wd1, 'bd1', bd1, 'Wd_output', Wd_output, 'bd_output', bd_output), X_train, relu);
            fprintf('  Epoch %d: Loss = %.6f\n', epoch, current_loss);
        end
    end
    
    % Return final weights
    final_weights = struct('We1', We1, 'be1', be1, 'We_latent', We_latent, 'be_latent', be_latent, 'Wd1', Wd1, 'bd1', bd1, 'Wd_output', Wd_output, 'bd_output', bd_output);
end

function loss = evaluate_final_loss(weights, X_train, relu_func)
    total_loss = 0;
    num_samples = size(X_train, 1);
    
    for i = 1:num_samples
        X_sample = X_train(i, :);
        [~, ~, sample_loss] = forward_pass(X_sample, weights.We1, weights.be1, weights.We_latent, weights.be_latent, weights.Wd1, weights.bd1, weights.Wd_output, weights.bd_output, relu_func);
        total_loss = total_loss + sample_loss;
    end
    
    loss = total_loss / num_samples;
end

function [Z_activated, X_reconstructed, sample_loss] = forward_pass(X_sample, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu)
    % Encoder Path
    H1_enc_linear = X_sample * We1 + be1;
    H1_enc_activated = relu(H1_enc_linear);

    Z_linear = H1_enc_activated * We_latent + be_latent;
    Z_activated = relu(Z_linear);

    % Decoder Path
    H1_dec_linear = Z_activated * Wd1 + bd1;
    H1_dec_activated = relu(H1_dec_linear);

    X_reconstructed_linear = H1_dec_activated * Wd_output + bd_output;
    X_reconstructed = X_reconstructed_linear;

    % Calculate Loss
    reconstruction_error = X_reconstructed - X_sample;
    sample_loss = 0.5 * sum(reconstruction_error.^2);
end

function [grad_We1, grad_be1, grad_We_latent, grad_be_latent, grad_Wd1, grad_bd1, grad_Wd_output, grad_bd_output] = backward_pass(X_sample, X_reconstructed, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu, relu_derivative)
    % Forward pass to get activations
    H1_enc_linear = X_sample * We1 + be1;
    H1_enc_activated = relu(H1_enc_linear);

    Z_linear = H1_enc_activated * We_latent + be_latent;
    Z_activated = relu(Z_linear);

    H1_dec_linear = Z_activated * Wd1 + bd1;
    H1_dec_activated = relu(H1_dec_linear);

    % Backwards pass
    reconstruction_error = X_reconstructed - X_sample;
    delta_output_layer = reconstruction_error;

    grad_Wd_output = H1_dec_activated' * delta_output_layer;
    grad_bd_output = delta_output_layer;

    delta_H1_dec_layer = (delta_output_layer * Wd_output') .* relu_derivative(H1_dec_activated);

    grad_Wd1 = Z_activated' * delta_H1_dec_layer;
    grad_bd1 = delta_H1_dec_layer;

    delta_Z_layer = (delta_H1_dec_layer * Wd1') .* relu_derivative(Z_activated);

    grad_We_latent = H1_enc_activated' * delta_Z_layer;
    grad_be_latent = delta_Z_layer;
    
    delta_H1_enc_layer = (delta_Z_layer * We_latent') .* relu_derivative(H1_enc_activated);
    
    grad_We1 = X_sample' * delta_H1_enc_layer;
    grad_be1 = delta_H1_enc_layer;
end