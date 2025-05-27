clear;clc;
load('data_1s.mat')

n = 10; % number of chunks
chunk_size = 384; % number of samples in a chunk
n_channel = 384; % number of channels to process
raw = data(1:n*chunk_size,1:n_channel);

%% step 1: Delta encoding
delta = diff(raw); % take the differential across rows

%% step 2: 3D plot before and after delta encoding
close all
figure('Position',[200,200,500,200]);

Z = raw;
% Create X and Y grid based on indices
[X,Y] = meshgrid(1:size(Z,2),1:size(Z,1));

% Plot surface
subplot(1,2,1);
surf(X,Y,double(Z),'EdgeColor','none'); 
xlabel('Channels')
ylabel('Samples')
zlabel('Voltage')
title('Raw data')
view(3)
zlim([-3000,3000])

Z = delta;
% Create X and Y grid based on indices
[X,Y] = meshgrid(1:size(Z,2),1:size(Z,1));

% Plot surface
subplot(1,2,2);
surf(X,Y,double(Z),'EdgeColor','none'); 
xlabel('Channels')
ylabel('Samples')
zlabel('Voltage')
title('Delta voltage across samples')
view(3)
zlim([-3000,3000])

%% Step 3: Auto-encoding
% --- User-Defined Configuration ---
learning_rate = 0.0002; % Base learning rate
num_epochs = 200;
latent_size_user = 200; 
% Current architecture: Input(384) -> Enc_H1(100) -> Latent(200) -> Dec_H1(100) -> Output(384)
enc_h1_size = 200; 
dec_h1_size = 200; 



% Take a subset of delta as training set (e.g., first 384 samples of delta)
% Ensure delta has enough rows. After diff, delta has size(raw,1)-1 rows.
num_train_samples_to_take = 384;
if size(delta,1) < num_train_samples_to_take
    error('Delta data has fewer rows than num_train_samples_to_take. Adjust selection or data size.');
end
X_train_original = delta(1:num_train_samples_to_take, :);
X_train_original = double(X_train_original);

% --- IMPLEMENTATION: 1. Data Normalization (Z-score) ---
fprintf('Normalizing X_train data (Z-score)...\n');
mean_X = mean(X_train_original, 1); % Calculate mean for each feature (column)
std_X = std(X_train_original, 0, 1);  % Calculate std for each feature (column)
% Replace std_X == 0 with 1 (or a small epsilon) to avoid division by zero if a feature is constant
std_X(std_X == 0) = 1e-6; % Using a small epsilon to avoid issues if std is exactly 0
X_train = (X_train_original - mean_X) ./ std_X;
fprintf('X_train has been Z-score normalized.\n');
% Note: To reconstruct to original scale later: X_reconstructed_orig = X_reconstructed_norm .* std_X + mean_X;
% -----------------------------------------------------------------------

% --- Determine Data Dimensions (from potentially normalized X_train) ---
input_size = size(X_train, 2);    % Number of features per sample
num_samples = size(X_train, 1);   % Total number of training samples

% --- Define Deep Network Layer Sizes ---
latent_size = latent_size_user;

fprintf('Network Configuration:\n');
fprintf('  Input Size: %d\n', input_size);
fprintf('  Encoder Hidden Layer 1 Size: %d\n', enc_h1_size);
fprintf('  Latent Space Size: %d\n', latent_size);
fprintf('  Decoder Hidden Layer 1 Size: %d\n', dec_h1_size);
fprintf('  Output Size: %d\n', input_size);
fprintf('  Number of Training Samples: %d\n\n', num_samples);

% --- 1. Initialize Deep Autoencoder Parameters (Using He Initialization for ReLU) ---
fprintf('Initializing weights and biases for DEEP autoencoder (He init)...\n');
% Encoder
% Layer 1: Input -> Encoder Hidden Layer 1
We1 = randn(input_size, enc_h1_size) * sqrt(2 / input_size);
be1 = zeros(1, enc_h1_size);
% Layer 2: Encoder Hidden Layer 1 -> Latent Space
We_latent = randn(enc_h1_size, latent_size) * sqrt(2 / enc_h1_size);
be_latent = zeros(1, latent_size);

% Decoder
% Layer 1: Latent Space -> Decoder Hidden Layer 1
Wd1 = randn(latent_size, dec_h1_size) * sqrt(2 / latent_size);
bd1 = zeros(1, dec_h1_size);
% Layer 2: Decoder Hidden Layer 1 -> Output Reconstruction
Wd_output = randn(dec_h1_size, input_size) * sqrt(2 / dec_h1_size);
bd_output = zeros(1, input_size);
fprintf('Initialization complete.\n\n');

% --- IMPLEMENTATION: 5. Activation Functions (ReLU for Hidden Layers) ---
relu = @(x) max(0, x);
relu_derivative = @(a) double(a > 0); % Derivative of ReLU w.r.t its input 'x', where 'a' is relu(x)
% Output layer will remain linear.
% -----------------------------------------------------------------------

% --- Function to perform forward pass ---
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

% --- Function to perform backward pass ---
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

%% --- Optimization Method Selection ---
% Choose one of: 'sgd', 'momentum', 'adagrad', 'adam'
optimization_method = 'adam';

% Hyperparameters for optimization methods
momentum_beta = 0.9;           % Momentum coefficient
adagrad_epsilon = 1e-8;        % AdaGrad stability constant
adam_beta1 = 0.9;              % Adam first moment coefficient
adam_beta2 = 0.999;            % Adam second moment coefficient
adam_epsilon = 1e-8;           % Adam stability constant

% --- Initialize optimizer-specific variables ---
% For Momentum
m_We1 = zeros(size(We1));
m_be1 = zeros(size(be1));
m_We_latent = zeros(size(We_latent));
m_be_latent = zeros(size(be_latent));
m_Wd1 = zeros(size(Wd1));
m_bd1 = zeros(size(bd1));
m_Wd_output = zeros(size(Wd_output));
m_bd_output = zeros(size(bd_output));

% For AdaGrad
cache_We1 = zeros(size(We1));
cache_be1 = zeros(size(be1));
cache_We_latent = zeros(size(We_latent));
cache_be_latent = zeros(size(be_latent));
cache_Wd1 = zeros(size(Wd1));
cache_bd1 = zeros(size(bd1));
cache_Wd_output = zeros(size(Wd_output));
cache_bd_output = zeros(size(bd_output));

% For Adam
m_We1_adam = zeros(size(We1));
v_We1_adam = zeros(size(We1));
m_be1_adam = zeros(size(be1));
v_be1_adam = zeros(size(be1));
m_We_latent_adam = zeros(size(We_latent));
v_We_latent_adam = zeros(size(We_latent));
m_be_latent_adam = zeros(size(be_latent));
v_be_latent_adam = zeros(size(be_latent));
m_Wd1_adam = zeros(size(Wd1));
v_Wd1_adam = zeros(size(Wd1));
m_bd1_adam = zeros(size(bd1));
v_bd1_adam = zeros(size(bd1));
m_Wd_output_adam = zeros(size(Wd_output));
v_Wd_output_adam = zeros(size(Wd_output));
m_bd_output_adam = zeros(size(bd_output));
v_bd_output_adam = zeros(size(bd_output));


%% --- 2. Training Loop ---
fprintf('Starting training with %s optimizer for %d epochs...\n', optimization_method, num_epochs);
loss_history = zeros(num_epochs, 1);
t = 0;  % Time step counter for Adam

for epoch = 1:num_epochs
    cumulative_epoch_loss = 0;
    X_train_shuffled = X_train(randperm(num_samples), :); % Shuffle samples each epoch

    for i = 1:num_samples
        t = t + 1;  % Increment time step for Adam
        X_sample = X_train_shuffled(i, :);

        % Forward pass
        [Z_activated, X_reconstructed, sample_loss] = forward_pass(X_sample, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu);
        cumulative_epoch_loss = cumulative_epoch_loss + sample_loss;

        % Backward pass
        [grad_We1, grad_be1, grad_We_latent, grad_be_latent, grad_Wd1, grad_bd1, grad_Wd_output, grad_bd_output] = ...
            backward_pass(X_sample, X_reconstructed, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu, relu_derivative);

        % Update weights based on selected optimization method
        switch optimization_method
            case 'sgd'
                % Standard SGD
                We1 = We1 - learning_rate * grad_We1;
                be1 = be1 - learning_rate * grad_be1;
                We_latent = We_latent - learning_rate * grad_We_latent;
                be_latent = be_latent - learning_rate * grad_be_latent;
                Wd1 = Wd1 - learning_rate * grad_Wd1;
                bd1 = bd1 - learning_rate * grad_bd1;
                Wd_output = Wd_output - learning_rate * grad_Wd_output;
                bd_output = bd_output - learning_rate * grad_bd_output;
                
            case 'momentum'
                % Momentum SGD
                m_We1 = momentum_beta * m_We1 + (1 - momentum_beta) * grad_We1;
                m_be1 = momentum_beta * m_be1 + (1 - momentum_beta) * grad_be1;
                m_We_latent = momentum_beta * m_We_latent + (1 - momentum_beta) * grad_We_latent;
                m_be_latent = momentum_beta * m_be_latent + (1 - momentum_beta) * grad_be_latent;
                m_Wd1 = momentum_beta * m_Wd1 + (1 - momentum_beta) * grad_Wd1;
                m_bd1 = momentum_beta * m_bd1 + (1 - momentum_beta) * grad_bd1;
                m_Wd_output = momentum_beta * m_Wd_output + (1 - momentum_beta) * grad_Wd_output;
                m_bd_output = momentum_beta * m_bd_output + (1 - momentum_beta) * grad_bd_output;
                
                We1 = We1 - learning_rate * m_We1;
                be1 = be1 - learning_rate * m_be1;
                We_latent = We_latent - learning_rate * m_We_latent;
                be_latent = be_latent - learning_rate * m_be_latent;
                Wd1 = Wd1 - learning_rate * m_Wd1;
                bd1 = bd1 - learning_rate * m_bd1;
                Wd_output = Wd_output - learning_rate * m_Wd_output;
                bd_output = bd_output - learning_rate * m_bd_output;
                
            case 'adagrad'
                % AdaGrad
                cache_We1 = cache_We1 + grad_We1.^2;
                cache_be1 = cache_be1 + grad_be1.^2;
                cache_We_latent = cache_We_latent + grad_We_latent.^2;
                cache_be_latent = cache_be_latent + grad_be_latent.^2;
                cache_Wd1 = cache_Wd1 + grad_Wd1.^2;
                cache_bd1 = cache_bd1 + grad_bd1.^2;
                cache_Wd_output = cache_Wd_output + grad_Wd_output.^2;
                cache_bd_output = cache_bd_output + grad_bd_output.^2;
                
                We1 = We1 - learning_rate * grad_We1 ./ (sqrt(cache_We1) + adagrad_epsilon);
                be1 = be1 - learning_rate * grad_be1 ./ (sqrt(cache_be1) + adagrad_epsilon);
                We_latent = We_latent - learning_rate * grad_We_latent ./ (sqrt(cache_We_latent) + adagrad_epsilon);
                be_latent = be_latent - learning_rate * grad_be_latent ./ (sqrt(cache_be_latent) + adagrad_epsilon);
                Wd1 = Wd1 - learning_rate * grad_Wd1 ./ (sqrt(cache_Wd1) + adagrad_epsilon);
                bd1 = bd1 - learning_rate * grad_bd1 ./ (sqrt(cache_bd1) + adagrad_epsilon);
                Wd_output = Wd_output - learning_rate * grad_Wd_output ./ (sqrt(cache_Wd_output) + adagrad_epsilon);
                bd_output = bd_output - learning_rate * grad_bd_output ./ (sqrt(cache_bd_output) + adagrad_epsilon);
                
            case 'adam'
                % Adam optimizer
                m_We1_adam = adam_beta1 * m_We1_adam + (1 - adam_beta1) * grad_We1;
                v_We1_adam = adam_beta2 * v_We1_adam + (1 - adam_beta2) * grad_We1.^2;
                m_be1_adam = adam_beta1 * m_be1_adam + (1 - adam_beta1) * grad_be1;
                v_be1_adam = adam_beta2 * v_be1_adam + (1 - adam_beta2) * grad_be1.^2;
                m_We_latent_adam = adam_beta1 * m_We_latent_adam + (1 - adam_beta1) * grad_We_latent;
                v_We_latent_adam = adam_beta2 * v_We_latent_adam + (1 - adam_beta2) * grad_We_latent.^2;
                m_be_latent_adam = adam_beta1 * m_be_latent_adam + (1 - adam_beta1) * grad_be_latent;
                v_be_latent_adam = adam_beta2 * v_be_latent_adam + (1 - adam_beta2) * grad_be_latent.^2;
                m_Wd1_adam = adam_beta1 * m_Wd1_adam + (1 - adam_beta1) * grad_Wd1;
                v_Wd1_adam = adam_beta2 * v_Wd1_adam + (1 - adam_beta2) * grad_Wd1.^2;
                m_bd1_adam = adam_beta1 * m_bd1_adam + (1 - adam_beta1) * grad_bd1;
                v_bd1_adam = adam_beta2 * v_bd1_adam + (1 - adam_beta2) * grad_bd1.^2;
                m_Wd_output_adam = adam_beta1 * m_Wd_output_adam + (1 - adam_beta1) * grad_Wd_output;
                v_Wd_output_adam = adam_beta2 * v_Wd_output_adam + (1 - adam_beta2) * grad_Wd_output.^2;
                m_bd_output_adam = adam_beta1 * m_bd_output_adam + (1 - adam_beta1) * grad_bd_output;
                v_bd_output_adam = adam_beta2 * v_bd_output_adam + (1 - adam_beta2) * grad_bd_output.^2;
                
                % Bias correction
                m_hat_We1 = m_We1_adam / (1 - adam_beta1^t);
                v_hat_We1 = v_We1_adam / (1 - adam_beta2^t);
                m_hat_be1 = m_be1_adam / (1 - adam_beta1^t);
                v_hat_be1 = v_be1_adam / (1 - adam_beta2^t);
                m_hat_We_latent = m_We_latent_adam / (1 - adam_beta1^t);
                v_hat_We_latent = v_We_latent_adam / (1 - adam_beta2^t);
                m_hat_be_latent = m_be_latent_adam / (1 - adam_beta1^t);
                v_hat_be_latent = v_be_latent_adam / (1 - adam_beta2^t);
                m_hat_Wd1 = m_Wd1_adam / (1 - adam_beta1^t);
                v_hat_Wd1 = v_Wd1_adam / (1 - adam_beta2^t);
                m_hat_bd1 = m_bd1_adam / (1 - adam_beta1^t);
                v_hat_bd1 = v_bd1_adam / (1 - adam_beta2^t);
                m_hat_Wd_output = m_Wd_output_adam / (1 - adam_beta1^t);
                v_hat_Wd_output = v_Wd_output_adam / (1 - adam_beta2^t);
                m_hat_bd_output = m_bd_output_adam / (1 - adam_beta1^t);
                v_hat_bd_output = v_bd_output_adam / (1 - adam_beta2^t);
                
                We1 = We1 - learning_rate * m_hat_We1 ./ (sqrt(v_hat_We1) + adam_epsilon);
                be1 = be1 - learning_rate * m_hat_be1 ./ (sqrt(v_hat_be1) + adam_epsilon);
                We_latent = We_latent - learning_rate * m_hat_We_latent ./ (sqrt(v_hat_We_latent) + adam_epsilon);
                be_latent = be_latent - learning_rate * m_hat_be_latent ./ (sqrt(v_hat_be_latent) + adam_epsilon);
                Wd1 = Wd1 - learning_rate * m_hat_Wd1 ./ (sqrt(v_hat_Wd1) + adam_epsilon);
                bd1 = bd1 - learning_rate * m_hat_bd1 ./ (sqrt(v_hat_bd1) + adam_epsilon);
                Wd_output = Wd_output - learning_rate * m_hat_Wd_output ./ (sqrt(v_hat_Wd_output) + adam_epsilon);
                bd_output = bd_output - learning_rate * m_hat_bd_output ./ (sqrt(v_hat_bd_output) + adam_epsilon);
                
            otherwise
                error('Unknown optimization method: %s', optimization_method);
        end
    end

    % Record average loss for this epoch
    loss_history(epoch) = cumulative_epoch_loss / num_samples;
    
    % Print progress
    if mod(epoch, 10) == 0 || epoch == 1 || epoch == num_epochs
        fprintf('Epoch %d/%d, Average Loss: %f\n', epoch, num_epochs, loss_history(epoch));
    end

end
fprintf('Training finished using %s optimizer.\n\n', optimization_method);

% --- 3. Plot Loss History ---
figure;
plot(1:num_epochs, loss_history);
xlabel('Epoch');
ylabel('Average Reconstruction Loss (Normalized Data)');
title(sprintf('Deep Autoencoder Training Loss - %s Optimizer', upper(optimization_method)));
grid on;

% --- 4. Test Reconstruction (on one original, then normalized sample) ---
if num_samples > 0
    test_sample_original_scale = X_train_original(1, :); % First sample from original, unnormalized training data
    
    % Normalize this test sample using the training set's mean and std
    test_sample_normalized = (test_sample_original_scale - mean_X) ./ std_X;
    
    % Forward pass with trained weights using the normalized sample
    [~, test_sample_reconstructed_normalized, mse_reconstruction] = forward_pass(test_sample_normalized, We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu);
    
    % De-normalize the reconstructed sample to compare with original scale
    test_sample_reconstructed_original_scale = test_sample_reconstructed_normalized .* std_X + mean_X;
    
    mse_reconstruction_original_scale = mean((test_sample_original_scale - test_sample_reconstructed_original_scale).^2);
    fprintf('MSE for reconstructing the first training sample (original scale): %f\n', mse_reconstruction_original_scale);
    
    % Plotting original vs reconstructed in original scale
    figure;
    subplot(1,2,1); plot(test_sample_original_scale); title('Original First Sample (Original Scale)'); axis tight;
    subplot(1,2,2); plot(test_sample_reconstructed_original_scale); title('Reconstructed First Sample (Original Scale)'); axis tight;
    sgtitle(sprintf('Sample Reconstruction Comparison - %s Optimizer', upper(optimization_method)));
else
    fprintf('No samples in X_train to test reconstruction.\n');
end
fprintf('Deep autoencoder %s optimization example complete.\n', optimization_method);

%% step 5: 3D plot of a subset of X_train_original and its reconstruction
fprintf('Reconstructing X_train_original for visualization...\n');

% Reconstruct X_train_original using the trained model
X_to_reconstruct = X_train_original;
% X_to_reconstruct = double(delta(384*3+1:384*4,:));
data_to_reconstruct_norm = (X_to_reconstruct - mean_X) ./ std_X;

% Forward pass for all samples
num_samples_total = size(data_to_reconstruct_norm, 1);
reconstructed_data_norm = zeros(num_samples_total, input_size);

for i = 1:num_samples_total
    [~, reconstructed_sample, ~] = forward_pass(data_to_reconstruct_norm(i,:), We1, be1, We_latent, be_latent, Wd1, bd1, Wd_output, bd_output, relu);
    reconstructed_data_norm(i,:) = reconstructed_sample;
end

% De-normalize the full reconstructed data to original scale
reconstructed_data_original_scale = reconstructed_data_norm .* std_X + mean_X;

% Visualization
fprintf('Plotting comparison of original and reconstructed X_train subset...\n');
figure('Position',[200,200,1200,500]);

% Define the number of samples to plot in the subset
num_samples_to_plot = min(size(X_to_reconstruct,1), 100); % Plot up to 100 samples

% Prepare subset of original data (unnormalized)
Z_before_subset = X_to_reconstruct(1:num_samples_to_plot, :);
[X_viz_before, Y_viz_before] = meshgrid(1:size(Z_before_subset,2), 1:size(Z_before_subset,1));

% Prepare corresponding subset of reconstructed data (de-normalized)
Z_after_subset = reconstructed_data_original_scale(1:num_samples_to_plot, :);
[X_viz_after, Y_viz_after] = meshgrid(1:size(Z_after_subset,2), 1:size(Z_after_subset,1));

% Determine common Z-limits for consistent scaling
common_z_min = min([min(Z_before_subset(:)), min(Z_after_subset(:))]);
common_z_max = max([max(Z_before_subset(:)), max(Z_after_subset(:))]);
if common_z_min == common_z_max
    common_z_min = common_z_min - 1;
    common_z_max = common_z_max + 1;
elseif (common_z_max - common_z_min) < 1e-6
    buffer = max(1, abs(common_z_min)*0.1);
    common_z_min = common_z_min - buffer;
    common_z_max = common_z_max + buffer;
end
if isnan(common_z_min) || isinf(common_z_min) || isnan(common_z_max) || isinf(common_z_max)
    warning('Could not determine valid common Z-limits. Plots will autoscale Z-axis.');
    use_common_zlim = false;
else
    use_common_zlim = true;
end

% Plot original X_train_original subset (unnormalized)
subplot(1,2,1);
surf(X_viz_before, Y_viz_before, double(Z_before_subset), 'EdgeColor', 'none'); 
xlabel('Channels'); 
ylabel(sprintf('Samples (1 to %d)', num_samples_to_plot)); 
zlabel('Voltage');
title('Original Delta (Before Auto-encoding)'); 
view(3); 
axis tight;
if use_common_zlim
    zlim([common_z_min, common_z_max]);
end

% Plot reconstructed X_train_original subset (de-normalized)
subplot(1,2,2);
surf(X_viz_after, Y_viz_after, double(Z_after_subset), 'EdgeColor', 'none'); 
xlabel('Channels'); 
ylabel(sprintf('Samples (1 to %d)', num_samples_to_plot)); 
zlabel('Voltage');
title('Reconstructed Delta (After Auto-encoding)'); 
view(3); 
axis tight;
if use_common_zlim
    zlim([common_z_min, common_z_max]);
end

sgtitle(sprintf('Comparison: Original vs. Reconstructed Data - %s Optimizer', upper(optimization_method)));

%% Comparative Analysis with Multiple Optimizers
% This section allows you to compare multiple optimizers

% Set to true to run the comparative analysis
run_comparison = false;

if run_comparison
    % Optimization methods to compare
    optimizers = {'sgd', 'momentum', 'adagrad', 'adam'};
    num_optimizers = length(optimizers);
    
    % Epochs for comparison (use fewer for quicker test)
    comparison_epochs = 300;
    
    % Store loss histories for each optimizer
    comparison_loss_history = zeros(comparison_epochs, num_optimizers);
    final_mse_values = zeros(1, num_optimizers);
    
    figure('Position', [200, 200, 1200, 600]);
    
    % Main loop to train with each optimizer
    for opt_idx = 1:num_optimizers
        current_optimizer = optimizers{opt_idx};
        fprintf('\n=== Running comparison with %s optimizer ===\n', upper(current_optimizer));
        
        % Reset network weights for fair comparison
        % Encoder
        We1_comp = randn(input_size, enc_h1_size) * sqrt(2 / input_size);
        be1_comp = zeros(1, enc_h1_size);
        We_latent_comp = randn(enc_h1_size, latent_size) * sqrt(2 / enc_h1_size);
        be_latent_comp = zeros(1, latent_size);
        
        % Decoder
        Wd1_comp = randn(latent_size, dec_h1_size) * sqrt(2 / latent_size);
        bd1_comp = zeros(1, dec_h1_size);
        Wd_output_comp = randn(dec_h1_size, input_size) * sqrt(2 / dec_h1_size);
        bd_output_comp = zeros(1, input_size);
        
        % Reset optimizer variables
        % For Momentum
        m_We1_comp = zeros(size(We1_comp));
        m_be1_comp = zeros(size(be1_comp));
        m_We_latent_comp = zeros(size(We_latent_comp));
        m_be_latent_comp = zeros(size(be_latent_comp));
        m_Wd1_comp = zeros(size(Wd1_comp));
        m_bd1_comp = zeros(size(bd1_comp));
        m_Wd_output_comp = zeros(size(Wd_output_comp));
        m_bd_output_comp = zeros(size(bd_output_comp));
        
        % For AdaGrad
        cache_We1_comp = zeros(size(We1_comp));
        cache_be1_comp = zeros(size(be1_comp));
        cache_We_latent_comp = zeros(size(We_latent_comp));
        cache_be_latent_comp = zeros(size(be_latent_comp));
        cache_Wd1_comp = zeros(size(Wd1_comp));
        cache_bd1_comp = zeros(size(bd1_comp));
        cache_Wd_output_comp = zeros(size(Wd_output_comp));
        cache_bd_output_comp = zeros(size(bd_output_comp));
        
        % For Adam
        m_We1_adam_comp = zeros(size(We1_comp));
        v_We1_adam_comp = zeros(size(We1_comp));
        m_be1_adam_comp = zeros(size(be1_comp));
        v_be1_adam_comp = zeros(size(be1_comp));
        m_We_latent_adam_comp = zeros(size(We_latent_comp));
        v_We_latent_adam_comp = zeros(size(We_latent_comp));
        m_be_latent_adam_comp = zeros(size(be_latent_comp));
        v_be_latent_adam_comp = zeros(size(be_latent_comp));
        m_Wd1_adam_comp = zeros(size(Wd1_comp));
        v_Wd1_adam_comp = zeros(size(Wd1_comp));
        m_bd1_adam_comp = zeros(size(bd1_comp));
        v_bd1_adam_comp = zeros(size(bd1_comp));
        m_Wd_output_adam_comp = zeros(size(Wd_output_comp));
        v_Wd_output_adam_comp = zeros(size(Wd_output_comp));
        m_bd_output_adam_comp = zeros(size(bd_output_comp));
        v_bd_output_adam_comp = zeros(size(bd_output_comp));
        
        % Time step counter for Adam
        t_comp = 0;
        
        % Training Loop
        for epoch = 1:comparison_epochs
            cumulative_epoch_loss = 0;
            X_train_shuffled = X_train(randperm(num_samples), :);
            
            for i = 1:num_samples
                t_comp = t_comp + 1;
                X_sample = X_train_shuffled(i, :);
                
                % Forward pass
                [Z_activated, X_reconstructed, sample_loss] = forward_pass(X_sample, We1_comp, be1_comp, We_latent_comp, be_latent_comp, Wd1_comp, bd1_comp, Wd_output_comp, bd_output_comp, relu);
                cumulative_epoch_loss = cumulative_epoch_loss + sample_loss;
                
                % Backward pass
                [grad_We1, grad_be1, grad_We_latent, grad_be_latent, grad_Wd1, grad_bd1, grad_Wd_output, grad_bd_output] = ...
                    backward_pass(X_sample, X_reconstructed, We1_comp, be1_comp, We_latent_comp, be_latent_comp, Wd1_comp, bd1_comp, Wd_output_comp, bd_output_comp, relu, relu_derivative);
                
                % Update weights based on selected optimization method
                switch current_optimizer
                    case 'sgd'
                        % Standard SGD
                        We1_comp = We1_comp - learning_rate * grad_We1;
                        be1_comp = be1_comp - learning_rate * grad_be1;
                        We_latent_comp = We_latent_comp - learning_rate * grad_We_latent;
                        be_latent_comp = be_latent_comp - learning_rate * grad_be_latent;
                        Wd1_comp = Wd1_comp - learning_rate * grad_Wd1;
                        bd1_comp = bd1_comp - learning_rate * grad_bd1;
                        Wd_output_comp = Wd_output_comp - learning_rate * grad_Wd_output;
                        bd_output_comp = bd_output_comp - learning_rate * grad_bd_output;
                        
                    case 'momentum'
                        % Momentum SGD
                        m_We1_comp = momentum_beta * m_We1_comp + (1 - momentum_beta) * grad_We1;
                        m_be1_comp = momentum_beta * m_be1_comp + (1 - momentum_beta) * grad_be1;
                        m_We_latent_comp = momentum_beta * m_We_latent_comp + (1 - momentum_beta) * grad_We_latent;
                        m_be_latent_comp = momentum_beta * m_be_latent_comp + (1 - momentum_beta) * grad_be_latent;
                        m_Wd1_comp = momentum_beta * m_Wd1_comp + (1 - momentum_beta) * grad_Wd1;
                        m_bd1_comp = momentum_beta * m_bd1_comp + (1 - momentum_beta) * grad_bd1;
                        m_Wd_output_comp = momentum_beta * m_Wd_output_comp + (1 - momentum_beta) * grad_Wd_output;
                        m_bd_output_comp = momentum_beta * m_bd_output_comp + (1 - momentum_beta) * grad_bd_output;
                        
                        We1_comp = We1_comp - learning_rate * m_We1_comp;
                        be1_comp = be1_comp - learning_rate * m_be1_comp;
                        We_latent_comp = We_latent_comp - learning_rate * m_We_latent_comp;
                        be_latent_comp = be_latent_comp - learning_rate * m_be_latent_comp;
                        Wd1_comp = Wd1_comp - learning_rate * m_Wd1_comp;
                        bd1_comp = bd1_comp - learning_rate * m_bd1_comp;
                        Wd_output_comp = Wd_output_comp - learning_rate * m_Wd_output_comp;
                        bd_output_comp = bd_output_comp - learning_rate * m_bd_output_comp;
                        
                    case 'adagrad'
                        % AdaGrad
                        cache_We1_comp = cache_We1_comp + grad_We1.^2;
                        cache_be1_comp = cache_be1_comp + grad_be1.^2;
                        cache_We_latent_comp = cache_We_latent_comp + grad_We_latent.^2;
                        cache_be_latent_comp = cache_be_latent_comp + grad_be_latent.^2;
                        cache_Wd1_comp = cache_Wd1_comp + grad_Wd1.^2;
                        cache_bd1_comp = cache_bd1_comp + grad_bd1.^2;
                        cache_Wd_output_comp = cache_Wd_output_comp + grad_Wd_output.^2;
                        cache_bd_output_comp = cache_bd_output_comp + grad_bd_output.^2;
                        
                        We1_comp = We1_comp - learning_rate * grad_We1 ./ (sqrt(cache_We1_comp) + adagrad_epsilon);
                        be1_comp = be1_comp - learning_rate * grad_be1 ./ (sqrt(cache_be1_comp) + adagrad_epsilon);
                        We_latent_comp = We_latent_comp - learning_rate * grad_We_latent ./ (sqrt(cache_We_latent_comp) + adagrad_epsilon);
                        be_latent_comp = be_latent_comp - learning_rate * grad_be_latent ./ (sqrt(cache_be_latent_comp) + adagrad_epsilon);
                        Wd1_comp = Wd1_comp - learning_rate * grad_Wd1 ./ (sqrt(cache_Wd1_comp) + adagrad_epsilon);
                        bd1_comp = bd1_comp - learning_rate * grad_bd1 ./ (sqrt(cache_bd1_comp) + adagrad_epsilon);
                        Wd_output_comp = Wd_output_comp - learning_rate * grad_Wd_output ./ (sqrt(cache_Wd_output_comp) + adagrad_epsilon);
                        bd_output_comp = bd_output_comp - learning_rate * grad_bd_output ./ (sqrt(cache_bd_output_comp) + adagrad_epsilon);
                        
                    case 'adam'
                        % Adam optimizer
                        m_We1_adam_comp = adam_beta1 * m_We1_adam_comp + (1 - adam_beta1) * grad_We1;
                        v_We1_adam_comp = adam_beta2 * v_We1_adam_comp + (1 - adam_beta2) * grad_We1.^2;
                        m_be1_adam_comp = adam_beta1 * m_be1_adam_comp + (1 - adam_beta1) * grad_be1;
                        v_be1_adam_comp = adam_beta2 * v_be1_adam_comp + (1 - adam_beta2) * grad_be1.^2;
                        m_We_latent_adam_comp = adam_beta1 * m_We_latent_adam_comp + (1 - adam_beta1) * grad_We_latent;
                        v_We_latent_adam_comp = adam_beta2 * v_We_latent_adam_comp + (1 - adam_beta2) * grad_We_latent.^2;
                        m_be_latent_adam_comp = adam_beta1 * m_be_latent_adam_comp + (1 - adam_beta1) * grad_be_latent;
                        v_be_latent_adam_comp = adam_beta2 * v_be_latent_adam_comp + (1 - adam_beta2) * grad_be_latent.^2;
                        m_Wd1_adam_comp = adam_beta1 * m_Wd1_adam_comp + (1 - adam_beta1) * grad_Wd1;
                        v_Wd1_adam_comp = adam_beta2 * v_Wd1_adam_comp + (1 - adam_beta2) * grad_Wd1.^2;
                        m_bd1_adam_comp = adam_beta1 * m_bd1_adam_comp + (1 - adam_beta1) * grad_bd1;
                        v_bd1_adam_comp = adam_beta2 * v_bd1_adam_comp + (1 - adam_beta2) * grad_bd1.^2;
                        m_Wd_output_adam_comp = adam_beta1 * m_Wd_output_adam_comp + (1 - adam_beta1) * grad_Wd_output;
                        v_Wd_output_adam_comp = adam_beta2 * v_Wd_output_adam_comp + (1 - adam_beta2) * grad_Wd_output.^2;
                        m_bd_output_adam_comp = adam_beta1 * m_bd_output_adam_comp + (1 - adam_beta1) * grad_bd_output;
                        v_bd_output_adam_comp = adam_beta2 * v_bd_output_adam_comp + (1 - adam_beta2) * grad_bd_output.^2;
                        
                        % Bias correction
                        m_hat_We1 = m_We1_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_We1 = v_We1_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_be1 = m_be1_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_be1 = v_be1_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_We_latent = m_We_latent_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_We_latent = v_We_latent_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_be_latent = m_be_latent_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_be_latent = v_be_latent_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_Wd1 = m_Wd1_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_Wd1 = v_Wd1_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_bd1 = m_bd1_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_bd1 = v_bd1_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_Wd_output = m_Wd_output_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_Wd_output = v_Wd_output_adam_comp / (1 - adam_beta2^t_comp);
                        m_hat_bd_output = m_bd_output_adam_comp / (1 - adam_beta1^t_comp);
                        v_hat_bd_output = v_bd_output_adam_comp / (1 - adam_beta2^t_comp);
                        
                        We1_comp = We1_comp - learning_rate * m_hat_We1 ./ (sqrt(v_hat_We1) + adam_epsilon);
                        be1_comp = be1_comp - learning_rate * m_hat_be1 ./ (sqrt(v_hat_be1) + adam_epsilon);
                        We_latent_comp = We_latent_comp - learning_rate * m_hat_We_latent ./ (sqrt(v_hat_We_latent) + adam_epsilon);
                        be_latent_comp = be_latent_comp - learning_rate * m_hat_be_latent ./ (sqrt(v_hat_be_latent) + adam_epsilon);
                        Wd1_comp = Wd1_comp - learning_rate * m_hat_Wd1 ./ (sqrt(v_hat_Wd1) + adam_epsilon);
                        bd1_comp = bd1_comp - learning_rate * m_hat_bd1 ./ (sqrt(v_hat_bd1) + adam_epsilon);
                        Wd_output_comp = Wd_output_comp - learning_rate * m_hat_Wd_output ./ (sqrt(v_hat_Wd_output) + adam_epsilon);
                        bd_output_comp = bd_output_comp - learning_rate * m_hat_bd_output ./ (sqrt(v_hat_bd_output) + adam_epsilon);
                end
            end
            
            % Record average loss for this epoch
            avg_epoch_loss = cumulative_epoch_loss / num_samples;
            comparison_loss_history(epoch, opt_idx) = avg_epoch_loss;
            
            % Print progress
            if mod(epoch, 50) == 0 || epoch == 1 || epoch == comparison_epochs
                fprintf('Epoch %d/%d, Average Loss: %f\n', epoch, comparison_epochs, avg_epoch_loss);
            end
        end
        
        % Calculate and store MSE for a test sample
        test_sample_normalized = (X_train_original(1, :) - mean_X) ./ std_X;
        [~, test_reconstruction_normalized, ~] = forward_pass(test_sample_normalized, We1_comp, be1_comp, We_latent_comp, be_latent_comp, Wd1_comp, bd1_comp, Wd_output_comp, bd_output_comp, relu);
        test_reconstruction_original = test_reconstruction_normalized .* std_X + mean_X;
        test_mse = mean((X_train_original(1, :) - test_reconstruction_original).^2);
        final_mse_values(opt_idx) = test_mse;
        
        fprintf('Final MSE for %s optimizer: %f\n', upper(current_optimizer), test_mse);
    end
    
    % Plot comparison of loss curves
    subplot(2, 2, [1, 2]);
    hold on;
    colors = {'b', 'r', 'g', 'm'};
    for i = 1:num_optimizers
        plot(1:comparison_epochs, comparison_loss_history(:, i), colors{i}, 'LineWidth', 2);
    end
    hold off;
    xlabel('Epoch');
    ylabel('Average Loss');
    title('Comparison of Optimization Methods');
    legend(upper(optimizers), 'Location', 'northeast');
    grid on;
    
    % Plot final MSE values as bar chart
    subplot(2, 2, 3);
    bar(final_mse_values);
    set(gca, 'XTickLabel', upper(optimizers));
    xlabel('Optimizer');
    ylabel('Final MSE');
    title('Final MSE Comparison');
    grid on;
    
    % Plot log-scale comparison for better visibility of differences
    subplot(2, 2, 4);
    hold on;
    for i = 1:num_optimizers
        semilogy(1:comparison_epochs, comparison_loss_history(:, i), colors{i}, 'LineWidth', 2);
    end
    hold off;
    xlabel('Epoch');
    ylabel('Loss (Log Scale)');
    title('Comparison (Log Scale)');
    legend(upper(optimizers), 'Location', 'northeast');
    grid on;
    
    sgtitle('Autoencoder Optimization Methods Comparison');
end

%% Function to compare specific optimizers with tuned hyperparameters
function compare_tuned_optimizers()
    % This function can be customized to compare specific optimizers with
    % different hyperparameter settings for your specific use case
    
    % Example optimization settings to compare:
    % 1. Adam with different learning rates
    % 2. Momentum with different beta values 
    % 3. AdaGrad with different epsilon values
    % etc.
    
    fprintf('To use this function, define your specific comparison parameters\n');
    fprintf('and implement the comparison logic similar to the main comparison loop.\n');
end