function [global_best_loss_history, weight_log, final_mse, p_best_particles_loss] = ...
    train_autoencoder_pso(X_train, param_template, relu, optimizer_type, num_epochs, num_particles, pso_params)

    % Flatten parameter struct to vector and get unpacking function
    % [init_vector, unpack] = flatten_params(param_template);
    % dim = numel(init_vector);
    % 
    % % === Initialize Particles and Velocities ===
    % particles = zeros(num_particles, dim);
    % rand_indices = randi(dim, num_particles, 1);
    % for i = 1:num_particles
    %     particles(i, rand_indices(i)) = 1;
    % end
    % velocities = randn(num_particles, dim) * 0.1;
    noise_added = 0;

    % === He Initialization Around a Realistic Start Point ===
    init_params = param_template;
    fields = fieldnames(init_params);
    for f = 1:numel(fields)
        key = fields{f};
        sz = size(init_params.(key));
        if contains(key, 'We')  % weight matrix
            fan_in = sz(1);  % He initialization uses input size
            init_params.(key) = randn(sz) * sqrt(2 / fan_in);
        else  % biases
            init_params.(key) = zeros(sz);
        end
    end
    
    % Flatten to vector
    [init_vector, unpack] = flatten_params(init_params);
    dim = numel(init_vector);
    if size(init_vector, 1) > 1 && size(init_vector, 2) == 1
        init_vector = init_vector';  % Make it a row vector [1 × dim]
    end


    % === Initialize Particles and Velocities ===
    particles = repmat(init_vector, num_particles, 1) + randn(num_particles, dim) * 0.1;
    velocities = randn(num_particles, dim) * 0.1;
    % particles = max(min(particles, 0.1), -0.1);


    % === Initialize PSO State ===
    p_best_particles = particles;
    p_best_particles_loss = inf(num_particles, 2);
    g_best_particle = particles(1, :);  % Will be overwritten
    particle_history = cell(num_epochs, 1);
    best_particle_indices = zeros(num_epochs, 1);
    g_best_vectors = zeros(num_epochs, dim);
    global_best_loss_history = nan(num_epochs, 1);
    weight_log.optimizer = optimizer_type;
    weight_log.epoch = cell(num_epochs, 1);
    lastStr = '';

    % === Begin Optimization Loop ===
    % progressBar = waitbar(0, 'Training (PSO)...');
    for epoch = 1:num_epochs
        if epoch == 1
            lastStr = '';
        end
    
        % Create the new training status message
        str = sprintf('Training: Epoch %d/%d with optimizer %s', epoch, num_epochs, optimizer_type);
    
        % Clear previous line using carriage return and space overwrite
        if ~isempty(lastStr)
            fprintf('\r%s\r', repmat(' ', 1, length(lastStr)));  % Overwrite old line
        end
        
        % Print new message
        fprintf('%s', str);  
        lastStr = str;

    
        % Remember message length for next iteration
        lastStr = str;
        batch_size = pso_params.batch_size;
        num_samples = size(X_train, 1);
        num_batches = ceil(num_samples / batch_size);
    
        for i = 1:num_particles % for every particle caluculate the batch-loss and update according to it
            particle_vector = particles(i, :);
            params = unpack(particle_vector);
    
            total_loss = 0;
    
            for b = 1:num_batches
                start_idx = (b - 1) * batch_size + 1;
                end_idx = min(b * batch_size, num_samples);
                X_batch = X_train(start_idx:end_idx, :);
    
                [batch_loss, ~, ~] = forward_backward_pass(X_batch, params, relu, []);
                total_loss = total_loss + batch_loss;
            end
    
            total_avg_loss = total_loss / num_batches;
    
            % === Regularization Terms ===
            reg_penalty = pso_params.lambda_out * sum((abs(particle_vector) > pso_params.threshold) .* (particle_vector.^2)) ...
                        + pso_params.lambda_div / (var(particle_vector) + 1e-6);
            fitness = total_avg_loss + reg_penalty;


            % === Update Personal Best ===
            if fitness < p_best_particles_loss(i)
                p_best_particles_loss(i,:) = [fitness; total_avg_loss];
                p_best_particles(i, :) = particle_vector;
            end
        end
        

        % === Update Global Best ===
        [~, best_idx] = min(p_best_particles_loss(:,1));  % Nur fitness-Spalte vergleichen
        global_best_loss = p_best_particles_loss(best_idx, :);  % Hol beide Werte (fitness & rec. loss)
        global_best_loss_history(epoch) = global_best_loss(2);  % Rekonstruktionsfehler loggen
        disp(global_best_loss(2))
        g_best_particle = p_best_particles(best_idx, :);
        best_particle_indices(epoch) = best_idx;
        g_best_vectors(epoch, :) = g_best_particle;
        weight_log.epoch{epoch} = g_best_particle;

        % === Log Particles and Update ===
        particle_history{epoch} = particles;
        [particles, velocities] = update_particles(particles, velocities, p_best_particles, g_best_particle, pso_params.w, pso_params.c1, pso_params.c2);
        particles = min(max(particles, -1), 1);

        % === Early Stopping ===
        if mean(std(particles)) < 1e-4 
            disp("Particles are converging – diversity is low.");
            x_train_sample = X_train(1,:);
            X_hats = reconstruction_over_all_epochs(x_train_sample, best_particle_indices, particle_history, unpack, relu);
            plot_particle_swarm_video_best_only(particle_history, g_best_vectors, x_train_sample, X_hats, global_best_loss_history, 'swarm_training.mp4');
            final_mse = compute_reconstruction_mse(params, X_train, relu);
            save_reconstruction_plot(x_train_sample, X_hats{epoch,:}, 'reconstruction_plot');
            return
        end
        % === Detect Early Convergence and Inject Noise Selectively ===
        if epoch > 2 && (abs(global_best_loss_history(epoch - 2) - global_best_loss_history(epoch)) < 1)
            % Only add noise every few epochs
            if noise_added < 1
                disp("add noise");
                noise_std = 0.1;
        
                % Create a mask for particles not equal to best_particle_indices
                not_best_mask = true(num_particles, 1);
                not_best_mask(best_particle_indices(epoch)) = false;
        
                % Inject noise only into non-best particles
                noise = randn(sum(not_best_mask), dim) * noise_std;
                particles(not_best_mask, :) = particles(not_best_mask, :) + noise;
                pso_params.c2 = 1.1;
                pso_params.w = 0.2;
                noise_added = 1;
            else
                noise_added = mod(noise_added + 1, 5);
            end
        end

        if epoch>10 && (abs(global_best_loss_history(epoch-10) -global_best_loss_history(epoch)) <1 )
            disp("Loss stagnating");
            x_train_sample = X_train(1,:);
            X_hats = reconstruction_over_all_epochs(x_train_sample, best_particle_indices, particle_history, unpack, relu);
            plot_particle_swarm_video_best_only(particle_history, g_best_vectors, x_train_sample, X_hats, global_best_loss_history, 'swarm_training.mp4');
            final_mse = compute_reconstruction_mse(params, X_train, relu);
            save_reconstruction_plot(x_train_sample, X_hats{epoch,:}, 'reconstruction_plot');
            return
        end
    end

    % === Final Evaluation and Visualization ===

            x_train_sample = X_train(1,:);
            X_hats = reconstruction_over_all_epochs(x_train_sample, best_particle_indices, particle_history, unpack, relu);
            plot_particle_swarm_video_best_only(particle_history, g_best_vectors, x_train_sample, X_hats, global_best_loss_history, 'swarm_training.mp4');
            final_mse = compute_reconstruction_mse(params, X_train, relu);
            save_reconstruction_plot(x_train_sample, X_hats{end,:}, 'reconstruction_plot');

    fprintf('\n');
end
