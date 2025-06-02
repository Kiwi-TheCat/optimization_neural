function plot_particle_swarm_video_best_only(particle_history, g_best_history, x_train_sample, X_hats, global_best_loss_history, filename)
    % Visualizes PSO particle behavior over epochs
    % Shows only the best reconstruction from the epoch with lowest loss

    if exist(filename, 'file')
        delete(filename);
    end

    num_epochs = numel(particle_history);
    [num_particles, dim] = size(particle_history{1});
    downsampled_idx = round(linspace(1, dim, min(300, dim)));

    % === Find the epoch with the best loss ===
    % === Video Setup ===
    v = VideoWriter(filename, 'MPEG-4');
    v.FrameRate = 10;
    open(v);

    f = figure('Name', 'Swarm Training Video (Best Reconstruction Only)', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 500]);

    for epoch = 1:num_epochs
        clf(f);
        t = tiledlayout(f, 1, 3, 'Padding', 'compact');
        if isempty(particle_history{epoch})
            %warning('No particle data at epoch %d', epoch);
            close(v);
            close(f);
            return;
        end
        % --- 1: Particle Positions (downsampled for visibility) ---
        nexttile(t, 1);
        particles = particle_history{epoch};
        cols = downsampled_idx(downsampled_idx <= size(particles, 2));
        particles = particle_history{epoch};
        cols = downsampled_idx(downsampled_idx <= size(particles, 2));  % ensure within bounds
        
        hold on;
        for i = 1:min(num_particles, 20)
            plot(cols, particles(i, cols), '-', 'LineWidth', 1);
        end
        if size(g_best_history, 1) >= epoch
            plot(cols, g_best_history(epoch, cols), 'k-', 'LineWidth', 2);
        end
        hold off;
        title(sprintf('Particle Positions - Epoch %d', epoch));
        xlabel('Parameter Index'); ylabel('Value');
        xlim([1, dim]); ylim padded; grid on; box on;

        % --- 2: Particle Deltas ---
        nexttile(t, 2);
        if epoch > 1
            delta = particles - particle_history{epoch - 1};
            delta = delta(:, cols);
            imagesc(delta);
            caxis([-0.05, 0.05]); colorbar;
            xlabel('Parameter Index'); ylabel('Particle #');
            title(sprintf('Delta Weights - Epoch %d', epoch));
            set(gca, 'YDir', 'normal');
        else
            axis off;
            text(0.5, 0.5, 'Initial epoch (no delta)', 'HorizontalAlignment', 'center', 'FontSize', 12);
        end

        % --- 3: Fixed Best Reconstruction ---
        nexttile(t, 3);
        plot(x_train_sample, 'b', 'LineWidth', 1.2); hold on;
        plot(X_hats{epoch}, 'r--', 'LineWidth', 1.2);
        legend('Original', 'Reconstructed');
        title('Best Particle Reconstruction');
        xlabel('Index'); ylabel('Voltage');
        axis tight; grid on;

        drawnow;
        frame = getframe(f);
        writeVideo(v, frame);
    end

    close(v);
    close(f);
end
