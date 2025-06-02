function [loss_history_after_each_update,loss_history_per_epoch, weight_log, total_avg_loss] = train_autoencoder(X_train_full, params, optim, relu, relu_deriv,...
    optimizer_type, learning_rate, num_epochs, regularization_lambda, batch_descend, batch_size,  lastStr)
    %TRAIN_AUTOENCODER Trains a shallow autoencoder using the given optimizer and training data.
    %
    %   Trains an autoencoder on normalized input data using gradient-based updates.
    %   The function supports flexible activation functions (relu and leaky_relu) and optimization algorithms.
    %
    %   Inputs:
    %       X_train         - (N x D) matrix of input samples (N samples, D features)
    %       params          - struct containing all network weights and biases
    %       optim           - struct containing optimizer state (e.g., Adam moments)
    %       relu            - function handle for the activation function
    %       relu_deriv      - function handle for the derivative of the activation function
    %       optimizer_type  - string, one of {'sgd', 'adagrad', 'adam'}
    %       learning_rate   - scalar learning rate
    %       num_epochs      - number of training epochs
    %
    %   Outputs:
    %       loss_history    - (num_epochs x 1) vector containing average loss per epoch
    %       weight_log      - struct containing snapshots of network weights over time
    %
    %   Notes:
    %       - The function stores snapshots of weights every 10 epochs in `weight_log`.
    %       - A waitbar is shown during training for progress indication.
    %       - Optionally calls `live_training_plot(...)` for visualizing reconstruction.


    % Initialize weight_log on first use
    weight_log.optimizer = optimizer_type;
    weight_log.epoch = [];  % will be filled later with matching struct
    num_samples = size(X_train_full, 1); % samples in one batch
   % progressBar = waitbar(0, 'Training...', 'WindowStyle', 'normal');
    if nargin < 12 || isempty(lastStr)
        lastStr = '';
    end

    % data = 30'000x385
    % batch_size = 300
    % evaluations = 30'000/300 = 10 -> 8 evaluations for the training
    % train/test = 20/80
    num_batches = ceil(num_samples / batch_size);
    loss_history_after_each_update = zeros(num_epochs*num_batches, 1);
    loss_history_per_epoch = zeros(num_epochs, 1);
    for epoch = 1:num_epochs % normally iterates over all the batches and 
        str = sprintf('Training: Epoch %d/%d with optimizer %s', epoch, num_epochs, optimizer_type);
        % Erase previous message using backspaces
        fprintf(repmat('\b', 1, length(lastStr)));
        
        % Print the new message
        fprintf('%s', str);
        
        % Store current message
        lastStr = str;
        total_batch_loss = 0;
        % Randomly shuffle full dataset at the start of each epoch
        idx = randperm(num_samples);
        X_shuffled = X_train_full(idx, :);
        % --- Training ---
        for i = 1:num_batches
            start_idx = (i - 1) * batch_size + 1;
            end_idx = min(i * batch_size, num_samples);
            X_batch = X_shuffled(start_idx:end_idx, :);

            if batch_descend
                [batch_loss, grads, ~] = forward_backward_pass(X_batch, params, relu, relu_deriv);
                [params, optim] = update_params(params, grads, optim, learning_rate, optimizer_type, epoch, regularization_lambda);
                total_batch_loss = total_batch_loss + batch_loss;
            else
                % sample-wise
                rng(epoch);  % reproducible shuffle of a batch
                idx = randperm(num_samples);
                X_batch_shuffled = X_batch(idx, :);
        
                for j = 1:size(X_batch_shuffled, 1)
                    x = X_batch_shuffled(j, :);
                    [loss, grads, ~] = forward_backward_pass(x, params, relu, relu_deriv);
                    total_batch_loss = total_batch_loss + loss;
                    [params, optim] = update_params(params, grads, optim, learning_rate, optimizer_type, j, regularization_lambda);
                end
            end
            if batch_descend
                loss_history_after_each_update((epoch-1)*num_batches + i) = batch_loss;
            else
                loss_history_after_each_update((epoch-1)*num_batches + num_batches) = total_batch_loss / num_samples;
            end
        end
        if batch_descend
            loss_history_per_epoch(epoch) = total_batch_loss / num_batches;
        else
            loss_history_per_epoch(epoch) = total_batch_loss / num_samples;
        end

        % Optional live plot 
        [x_test_sample, x_hat] = live_training_plot(X_train_full, params, epoch, relu); % takes the x_hat from last forward_backward_pass   



        % Save only every 10th epoch
        if mod(epoch, 10) == 0
            snapshot = save_epoch_weights(params);
            snapshot.epoch = epoch;
            snapshot.x_test_sample = x_test_sample;
            snapshot.x_hat = x_hat;

            if isempty(weight_log.epoch)
                weight_log.epoch = snapshot;  % first assignment defines struct fields
            else
                weight_log.epoch(end+1) = snapshot;  % safe append
            end

        end
    end
    total_avg_loss = total_batch_loss / num_batches;
    fprintf('\n');
end
