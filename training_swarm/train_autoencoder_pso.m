function [global_best_loss_history, weight_log, final_mse, p_best_particles_loss] = ...
    train_autoencoder_pso(X_train, param_template, relu, optimizer_type, num_epochs, num_particles, pso_params)

    % Flatten parameter struct to vector and get unpacking function
    [init_vector, unpack] = flatten_params(param_template);
    dim = numel(init_vector);

    % === Initialize Particles and Velocities ===
    particles = zeros(num_particles, dim);
    rand_indices = randi(dim, num_particles, 1);
    for i = 1:num_particles
        particles(i, rand_indices(i)) = 1;
    end
    velocities = randn(num_particles, dim) * 0.1;

    % === Initialize PSO State ===
    p_best_particles = particles;
    p_best_particles_loss = inf(num_particles, 1);
    g_best_particle = particles(1, :);  % Will be overwritten
    particle_history = cell(num_epochs, 1);
    best_particle_indices = zeros(num_epochs, 1);
    g_best_vectors = zeros(num_epochs, dim);
    global_best_loss_history = zeros(num_epochs, 1);
    weight_log.optimizer = optimizer_type;
    weight_log.epoch = cell(num_epochs, 1);
    lastStr = '';

    % === Begin Optimization Loop ===
    % progressBar = waitbar(0, 'Training (PSO)...');
    for epoch = 1:num_epochs
        str = sprintf('Training: Epoch %d/%d with optimizer %s', epoch, num_epochs, optimizer_type);
        % Erase previous message using backspaces
        fprintf(repmat('\b', 1, length(lastStr)));

        % === Evaluate Fitness for Each Particle ===
        for i = 1:num_particles
            particle_vector = particles(i, :);
            params = unpack(particle_vector);

            [total_avg_loss, ~, ~] = forward_backward_pass(X_train, params, relu, []); % the loss function, returns: gradients

            % === Regularization Terms ===
            reg_penalty = pso_params.lambda_out * sum((abs(particle_vector) > pso_params.threshold) .* (particle_vector.^2)) ...
                        + pso_params.lambda_div / (var(particle_vector) + 1e-6);
            fitness = total_avg_loss + reg_penalty;

            % === Update Personal Best ===
            if fitness < p_best_particles_loss(i)
                p_best_particles_loss(i) = fitness;
                p_best_particles(i, :) = particle_vector;
            end
        end

        % === Update Global Best ===
        [global_best_loss, best_idx] = min(p_best_particles_loss);
        g_best_particle = p_best_particles(best_idx, :);
        global_best_loss_history(epoch) = global_best_loss;
        best_particle_indices(epoch) = best_idx;
        g_best_vectors(epoch, :) = g_best_particle;
        weight_log.epoch{epoch} = g_best_particle;

        % === Log Particles and Update ===
        particle_history{epoch} = particles;
        [particles, velocities] = update_particles(...
            particles, velocities, p_best_particles, g_best_particle, ...
            pso_params.w, pso_params.c1, pso_params.c2);

        % === Early Stopping on Low Diversity ===
        if mean(std(particles)) < 1e-3
            disp("Particles are converging – diversity is low.");
            break;
        end
    end
    close(progressBar);

    % === Final Evaluation and Visualization ===
    params = unpack(g_best_particle);
    X_hats = reconstruction_over_all_epochs(X_train, best_particle_indices, particle_history, unpack, relu);
    plot_particle_swarm_video(particle_history, g_best_vectors, X_train(1,:), X_hats, 'swarm_training.mp4');
    final_mse = compute_reconstruction_mse(params, X_train, relu);
    fprintf('\n');
end
