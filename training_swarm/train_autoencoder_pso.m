function [loss_history, weight_log, final_mse, x_test_log_out, p_best_loss] = ...
    train_autoencoder_pso(X_train, X_train_original, mean_X, std_X, ...
    param_template, relu, optimizer_type, num_epochs, o)

    % So I make a mutliple of particles and then make them navigate the 
    % optimization by evaluating their loss after each update?
    % A particle x_i in R^((384+200)x1) represents a flat vector of all neural network parameters:
    
    % Option 1: At every swarm iteration, evaluate each particle’s loss on all signals, i.e., the full dataset.
    % Option 2: Each iteration, randomly pick a signal (or mini-batch), and evaluate fitness based on it.
    % Option 3: First optimize for signal 1, then 2, ..., then 𝑁, repeating the cycle until convergence.

    % === PSO parameters ===
    num_particles = 50;

    w = 0.5; % momentum
    c1 = 1.5; % cognitive
    c2 = 0.3; % social
    t = 0;

    % === Flatten initial parameter template ===
    [init_vector, unpack] = flatten_params(param_template);
    init_vector = reshape(init_vector, 1, []);  % ensure row vector
    dim = numel(init_vector);

    % === Initialize swarm around He-initialized vector ===
    %fan_in = dim;
    %he_std = sqrt(2 / fan_in);
    %particles = he_std * randn(num_particles, dim);

    % one hot particles
    particles = zeros(num_particles, dim);
    rand_indices = randi(dim, num_particles, 1);
    
    for i = 1:num_particles
        particles(i, rand_indices(i)) = 1;
    end
    
    % swarm initialisation with center at intialisation vector
    % particles = repmat(init_vector, num_particles, 1) + 0.1 * randn(num_particles, dim); 
    velocities = randn(num_particles, dim) * 0.1;

    % === Swarm state ===
    p_best = particles; % saves the personal best particle
    p_best_loss = inf(num_particles, 1); % saves the personal best loss
    g_best = particles(1, :); % saves the global best particle
    g_best_loss = inf; % saves the global best loss
    particle_history = cell(num_epochs, 1); % saves the particle history for visualization of the particle generations
    best_particle_indices = zeros(num_epochs, 1);       % index of best particle at each epoch
    %g_best_vectors = zeros(num_epochs, dim);            % g_best particle vector per epoch

    % === Logs ===
    loss_history = zeros(num_epochs, 1); % all the g_best loss
    weight_log.optimizer = optimizer_type;
    x_test_log_out = cell(num_epochs, 1); % x_test_log_out{epoch} = [x_test_sample; x_hat];
    progressBar = waitbar(0, 'Training (PSO)...');

    % === Begin PSO training loop ===
    for epoch = 1:num_epochs
        %w = w-0.003; % adds an decay to the weight of the previous velocity. Less momentum
        if isvalid(progressBar)
            waitbar(epoch / num_epochs, progressBar, sprintf('PSO Epoch %d/%d', epoch, num_epochs));
        end

        % === Evaluate all particles ===
        for i = 1:num_particles
            particle_vector = particles(i, :);
            params = unpack(particle_vector);
            mse_batch_loss = 0;
            % per particle evaluate all the signals from one batch and sum up their loss:= batch loss
            for n = 1:size(X_train, 1)
                x = X_train(n, :);
                [loss, ~, ~] = forward_backward_pass(x, params, relu, []);  % only forward
                mse_batch_loss = mse_batch_loss + loss;
            end

            mse_batch_loss = mse_batch_loss / size(X_train, 1);
            % Add regularization penalty on the particle vector
            lambda_out = 1e-3;   % penalize large individual weights (outliers)
            threshold   = 3;     % values above this are considered outliers
            lambda_div = 1e-3;   % penalize low diversity (low variance)

            reg_term = lambda_out * sum((abs(particle_vector) > threshold) .* (particle_vector.^2)) + lambda_div / (var(particle_vector) + 1e-6);
            mse_batch_loss = mse_batch_loss + reg_term;

            % === Update personal and global bests ===
            if mse_batch_loss < p_best_loss(i)
                p_best_loss(i) = mse_batch_loss;
                p_best(i, :) = particle_vector;
            end

            if mse_batch_loss < g_best_loss
                g_best_loss = mse_batch_loss;
                g_best = particle_vector;            
            end
        end  % END of particle evaluation loop

        [epoch_best_loss, best_idx] = min(p_best_loss);
        g_best = p_best(best_idx, :);
        g_best_loss = epoch_best_loss;
        
        % Log for later analysis
        best_particle_indices(epoch) = best_idx;
        %g_best_vectors(epoch, :) = g_best; % not yet logged

        % === Update all particles based on p_best and g_best ===
        [particles, velocities] = update_particles(particles, velocities, p_best, g_best, w, c1, c2);
        loss_history(epoch) = g_best_loss;
        particle_history{epoch} = particles;

        % === Save best model weights ===
        weight_log.epoch{epoch} = g_best;

        % === reconstruction of one sample in the batch ===
        params = unpack(g_best);
        x_test_sample = X_train_original(1, :);
        %x_norm = (x_test_sample - mean_X) ./ std_X;
        h1 = relu(x_test_sample * params.We1 + params.be1);
        z = relu(h1 * params.We_latent + params.be_latent);
        h2 = relu(z * params.Wd1 + params.bd1);
        x_hat = h2 * params.Wd_output + params.bd_output;
        x_hat = x_hat .* std_X + mean_X;
        x_test_log_out{epoch} = [x_test_sample; x_hat];

        % === Optional live plot ===
        %live_training_plot(x_test_log_out, optimizer_type, epoch);
        if mean(std(particles)) < 1e-3  
            disp("Particles are converging – diversity is low.")
            close(progressBar);
            plot_particle_swarm_video(particle_history, loss_history, x_test_log_out, 1, 'swarm_training.mp4');
            params = unpack(g_best);
            final_mse = compute_reconstruction_mse(params, X_train_original, mean_X, std_X, relu);      
            return;
        end
    end
    close(progressBar);

    plot_particle_swarm_video(particle_history, loss_history, x_test_log_out, 1, 'swarm_training.mp4');
    params = unpack(g_best);
    final_mse = compute_reconstruction_mse(params, X_train_original, mean_X, std_X, relu);

end
