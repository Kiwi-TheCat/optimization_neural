function [loss_history, weight_log, final_mse, x_test_log_out] = ...
    train_autoencoder(X_train, X_train_original, mean_X, std_X, ...
    params, optim, relu, relu_deriv, optimizer_type, learning_rate, num_epochs, o)

    num_samples = size(X_train, 1);
    loss_history = zeros(num_epochs, 1);
    t = 0;
    progressBar = waitbar(0, 'Training...', 'WindowStyle', 'normal');

    x_test_log_out = cell(num_epochs, 1);  % Log reconstructions

    % Initialize weight_log on first use
    weight_log.optimizer = optimizer_type;
    weight_log.epoch = [];  % will be filled later with matching struct

    for epoch = 1:num_epochs
        msg = sprintf('Training with %s (Epoch %d/%d)', upper(optimizer_type), epoch, num_epochs);
        waitbar(epoch / num_epochs, progressBar, msg);
        X_train = X_train(randperm(num_samples), :);
        total_loss = 0;

        % --- Training loop ---
        for i = 1:num_samples
            t = t + 1;
            x = X_train(i, :);
            [loss, grads, x_hat] = forward_backward_pass(x, params, relu, relu_deriv);
            total_loss = total_loss + loss;
            [params, optim] = update_params(params, grads, optim, learning_rate, optimizer_type, t);
        end

        loss_history(epoch) = total_loss / num_samples;

        % --- Test one reconstruction ---
        x_test_sample = X_train_original(1,:);
        x_norm = (x_test_sample - mean_X) ./ std_X;
        h1 = relu(x_norm * params.We1 + params.be1);
        z = relu(h1 * params.We_latent + params.be_latent);

        % Save only every 10th epoch
        if mod(epoch, 10) == 0
            snapshot = save_epoch_weights(params, z);
            snapshot.epoch = epoch;

            if isempty(weight_log.epoch)
                weight_log.epoch = snapshot;  % first assignment defines struct fields
            else
                weight_log.epoch(end+1) = snapshot;  % safe append
            end
        end

        % --- Final decoding for visualization ---
        h2 = relu(z * params.Wd1 + params.bd1);
        x_hat = h2 * params.Wd_output + params.bd_output;
        x_hat = x_hat .* std_X + mean_X;
        x_test_log_out{epoch} = [x_test_sample; x_hat];

        % Optional live plot if snapshot exists
        if exist('weight_log', 'var') && isfield(weight_log, 'epoch') && ~isempty(weight_log.epoch)
            live_training_plot(params, weight_log, X_train_original, mean_X, std_X, optimizer_type, epoch);
        end
    end

    % Compute final MSE
    final_mse = compute_reconstruction_mse(params, X_train_original, mean_X, std_X, relu);
    close(progressBar);
end
