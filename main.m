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
learning_rate = 0.0002; % You might need to tune this, especially with ReLU
num_epochs = 200;
latent_size_user = 200; 

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
% Current architecture: Input(384) -> Enc_H1(100) -> Latent(200) -> Dec_H1(100) -> Output(384)
enc_h1_size = 200; 
dec_h1_size = 200; 

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

% --- 2. Gradient Descent Training Loop ---
fprintf('Starting training with %d samples...\n', num_samples);
loss_history = zeros(num_epochs, 1);

for epoch = 1:num_epochs
    cumulative_epoch_loss = 0;
    X_train_shuffled = X_train(randperm(num_samples), :); % Shuffle samples each epoch

    for i = 1:num_samples
        X_sample = X_train_shuffled(i, :); 

        % --- 2a. Forward Propagation (with ReLU for hidden layers) ---
        % Encoder Path
        H1_enc_linear = X_sample * We1 + be1;
        H1_enc_activated = relu(H1_enc_linear); % ReLU activation

        Z_linear = H1_enc_activated * We_latent + be_latent;
        Z_activated = relu(Z_linear); % ReLU activation (Latent representation)

        % Decoder Path
        H1_dec_linear = Z_activated * Wd1 + bd1;
        H1_dec_activated = relu(H1_dec_linear); % ReLU activation

        X_reconstructed_linear = H1_dec_activated * Wd_output + bd_output;
        X_reconstructed = X_reconstructed_linear; % Linear activation for final output

        % --- 2b. Calculate Loss ---
        reconstruction_error = X_reconstructed - X_sample;
        sample_loss = 0.5 * sum(reconstruction_error.^2);
        cumulative_epoch_loss = cumulative_epoch_loss + sample_loss;

        % --- 2c. Backward Propagation (with ReLU derivatives) ---
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

        % --- 2d. Update Parameters ---
        We1 = We1 - learning_rate * grad_We1;
        be1 = be1 - learning_rate * grad_be1;
        We_latent = We_latent - learning_rate * grad_We_latent;
        be_latent = be_latent - learning_rate * grad_be_latent;

        Wd1 = Wd1 - learning_rate * grad_Wd1;
        bd1 = bd1 - learning_rate * grad_bd1;
        Wd_output = Wd_output - learning_rate * grad_Wd_output;
        bd_output = bd_output - learning_rate * grad_bd_output;
    end

    loss_history(epoch) = cumulative_epoch_loss / num_samples;
    if mod(epoch, 10) == 0 || epoch == 1 || epoch == num_epochs
        fprintf('Epoch %d/%d, Average Loss: %f\n', epoch, num_epochs, loss_history(epoch));
    end
end
fprintf('Training finished.\n\n');

% --- 3. Plot Loss History ---
figure;
plot(1:num_epochs, loss_history);
xlabel('Epoch');
ylabel('Average Reconstruction Loss (Normalized Data)');
title('Deep Autoencoder Training Loss Over Epochs');
grid on;

% --- 4. Test Reconstruction (on one original, then normalized sample) ---
if num_samples > 0
    test_sample_original_scale = X_train_original(1, :); % First sample from original, unnormalized training data
    
    % Normalize this test sample using the training set's mean and std
    test_sample_normalized = (test_sample_original_scale - mean_X) ./ std_X;
    
    % Forward pass with trained weights using the normalized sample
    H1_enc_linear_test = test_sample_normalized * We1 + be1;
    H1_enc_activated_test = relu(H1_enc_linear_test);
    Z_linear_test = H1_enc_activated_test * We_latent + be_latent;
    Z_activated_test = relu(Z_linear_test); % Encoded version
    H1_dec_linear_test = Z_activated_test * Wd1 + bd1;
    H1_dec_activated_test = relu(H1_dec_linear_test);
    X_reconstructed_linear_test = H1_dec_activated_test * Wd_output + bd_output;
    test_sample_reconstructed_normalized = X_reconstructed_linear_test; % Output is in normalized scale
    
    % De-normalize the reconstructed sample to compare with original scale
    test_sample_reconstructed_original_scale = test_sample_reconstructed_normalized .* std_X + mean_X;
    
    mse_reconstruction_original_scale = mean((test_sample_original_scale - test_sample_reconstructed_original_scale).^2);
    fprintf('MSE for reconstructing the first training sample (original scale): %f\n', mse_reconstruction_original_scale);
    
    % Plotting original vs reconstructed in original scale
    figure;
    subplot(1,2,1); plot(test_sample_original_scale); title('Original First Sample (Original Scale)'); axis tight;
    subplot(1,2,2); plot(test_sample_reconstructed_original_scale); title('Reconstructed First Sample (Original Scale)'); axis tight;
    sgtitle('Sample Reconstruction Comparison');

else
    fprintf('No samples in X_train to test reconstruction.\n');
end
fprintf('Deep autoencoder gradient descent example complete.\n');


%% step 5 (formerly step 4): 3D plot of a subset of X_train_original and its reconstruction
% This section uses X_train_original (unnormalized data that training was based on)
% for reconstruction, and then visualizes a subset in its original scale.

fprintf('Reconstructing X_train_original for visualization...\n');

% 1. Normalize X_train_original (the data that was used to train the AE)
%    using the mean_X and std_X derived during the training setup.
% X_to_reconstruct = double(delta(385:384*2,:));
X_to_reconstruct = X_train_original;
data_to_reconstruct_norm = (X_to_reconstruct - mean_X) ./ std_X;

% 2. Encoding Process using trained weights
H1_enc_linear_all = data_to_reconstruct_norm * We1 + be1;
H1_enc_activated_all = relu(H1_enc_linear_all);
Z_linear_all = H1_enc_activated_all * We_latent + be_latent;
Z_activated_all = relu(Z_linear_all); % Encoded representation

% 3. Decoding Process using trained weights
H1_dec_linear_all = Z_activated_all * Wd1 + bd1;
H1_dec_activated_all = relu(H1_dec_linear_all);
X_reconstructed_linear_all = H1_dec_activated_all * Wd_output + bd_output;
reconstructed_data_norm = X_reconstructed_linear_all; % Output is in normalized scale

% 4. De-normalize the full reconstructed data to original scale
reconstructed_data_original_scale = reconstructed_data_norm .* std_X + mean_X;

% 5. Visualization
fprintf('Plotting comparison of original and reconstructed X_train subset...\n');
figure('Position',[200,200,1200,500]); % Adjusted figure size for better viewing

% Define the number of samples to plot in the subset
num_samples_to_plot = min(size(X_to_reconstruct,1), 100); % Plot up to 100 samples

% Prepare subset of original data (unnormalized)
Z_before_subset = X_to_reconstruct(1:num_samples_to_plot, :);
[X_viz_before, Y_viz_before] = meshgrid(1:size(Z_before_subset,2), 1:size(Z_before_subset,1));

% Prepare corresponding subset of reconstructed data (de-normalized)
Z_after_subset = reconstructed_data_original_scale(1:num_samples_to_plot, :);
[X_viz_after, Y_viz_after] = meshgrid(1:size(Z_after_subset,2), 1:size(Z_after_subset,1));

% --- Determine common Z-limits for consistent scaling ---
common_z_min = min([min(Z_before_subset(:)), min(Z_after_subset(:))]);
common_z_max = max([max(Z_before_subset(:)), max(Z_after_subset(:))]);
% Add a small buffer to z-limits if they are too tight or identical
if common_z_min == common_z_max
    common_z_min = common_z_min - 1; % Adjust as needed
    common_z_max = common_z_max + 1; % Adjust as needed
elseif (common_z_max - common_z_min) < 1e-6 % If range is very small
    buffer = max(1, abs(common_z_min)*0.1); % Add a small buffer
    common_z_min = common_z_min - buffer;
    common_z_max = common_z_max + buffer;
end
if isnan(common_z_min) || isinf(common_z_min) || isnan(common_z_max) || isinf(common_z_max)
    warning('Could not determine valid common Z-limits. Plots will autoscale Z-axis.');
    use_common_zlim = false;
else
    use_common_zlim = true;
end
% --- End of Z-limit determination ---


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

sgtitle('Comparison: Original vs. Reconstructed Data Subset (Original Scale)');
