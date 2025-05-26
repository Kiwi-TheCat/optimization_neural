function [loss_history, weight_log, final_mse, x_test_log_out] = ...
    train_autoencoder(X_train, X_train_original, mean_X, std_X, ...
    params, optim, relu, relu_deriv, optimizer_type, learning_rate, num_epochs, o)

    num_samples = size(X_train, 1);
    loss_history = zeros(num_epochs, 1);
    t = 0;
    progressBar = waitbar(0, 'Training...', 'WindowStyle', 'normal');

    x_train_log_out = cell(num_epochs, 1);  % ← add this
    x_test_log_out = cell(num_epochs, 1);   % ← add this

    for epoch = 1:num_epochs
        msg = sprintf('Training with %s (Epoch %d/%d)', upper(optimizer_type), epoch, num_epochs);
        waitbar(epoch / num_epochs, progressBar, msg);
        X_train = X_train(randperm(num_samples), :);
        total_loss = 0;

        for i = 1:num_samples
            t = t + 1;
            x = X_train(i, :);
            [loss, grads, x_hat] = forward_backward_pass(x, params, relu, relu_deriv);
            total_loss = total_loss + loss;
            [params, optim] = update_params(params, grads, optim, learning_rate, optimizer_type, t);
        end

        loss_history(epoch) = total_loss / num_samples;
        weight_log.optimizer = optimizer_type;
        weight_log.epoch(epoch) = save_epoch_weights(params);

        % === Log reconstructions ===
        %x_train_log_out{epoch} = X_train;  % Log full training set (normalized)
        x_test_sample = X_train_original(1,:);
        x_norm = (x_test_sample - mean_X) ./ std_X;
        h1 = relu(x_norm * params.We1 + params.be1);
        z = relu(h1 * params.We_latent + params.be_latent);
        h2 = relu(z * params.Wd1 + params.bd1);
        x_hat = h2 * params.Wd_output + params.bd_output;
        x_hat = x_hat .* std_X + mean_X;
        x_test_log_out{epoch} = [x_test_sample; x_hat];  % original and reconstructed

        % Optional live plot
        live_training_plot(params, weight_log, X_train_original, mean_X, std_X, optimizer_type, epoch);
    end

    final_mse = compute_reconstruction_mse(params, X_train_original, mean_X, std_X, relu);
    close(progressBar);
end
