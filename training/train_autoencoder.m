function [loss_history, weight_log, total_avg_loss] = train_autoencoder(X_train, params, optim, relu, relu_deriv,...
    optimizer_type, learning_rate, num_epochs, regularization_lambda, batch_descend, lastStr)
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
    loss_history = zeros(num_epochs, 1);
    num_samples = size(X_train, 1); % samples in one batch
   % progressBar = waitbar(0, 'Training...', 'WindowStyle', 'normal');
    if nargin < 12 || isempty(lastStr)
        lastStr = '';
    end
    
    % data = 30'000x385
    % batch_size = 300
    % evaluations = 30'000/300 = 10 -> 8 evaluations for the training
    % train/test = 20/80
    for epoch = 1:num_epochs % normally iterates over all the batches and 
        str = sprintf('Training: Epoch %d/%d with optimizer %s', epoch, num_epochs, optimizer_type);
        % Erase previous message using backspaces
        fprintf(repmat('\b', 1, length(lastStr)));
        
        % Print the new message
        fprintf('%s', str);
        
        % Store current message
        lastStr = str;

        total_batch_loss = 0;
        % --- Training ---
        if batch_descend % one batch 384 samples 
            [total_avg_loss, grads, ~] = forward_backward_pass(X_train, params, relu, relu_deriv); % the loss function, returns: gradients
            [params, optim] = update_params(params, grads, optim, learning_rate, optimizer_type, epoch, regularization_lambda);
            loss_history(epoch) = total_avg_loss;
        else
            for i = 1:num_samples
                rng(epoch);  % Set once for reproducibility
                idx = randperm(num_samples);  % Will be the same every run
                X_train_shuffled = X_train(idx, :);
                x = X_train_shuffled(i, :);
                [loss, grads, ~] = forward_backward_pass(x, params, relu, relu_deriv); % the loss function, returns: gradients
                total_batch_loss = total_batch_loss + loss;
                [params, optim] = update_params(params, grads, optim, learning_rate, optimizer_type,i, regularization_lambda);
            end
            loss_history(epoch) = total_batch_loss / num_samples;
        end
        % Optional live plot 
        [x_test_sample, x_hat] = live_training_plot(X_train, params, epoch, relu); % takes the x_hat from last forward_backward_pass

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
    if ~batch_descend
        total_avg_loss = total_batch_loss / num_samples;
    end
    fprintf('\n');
end
