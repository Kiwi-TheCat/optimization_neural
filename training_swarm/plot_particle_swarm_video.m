function plot_particle_swarm_video(particle_history, g_best_history, x_train_selected, X_hats, filename)
    % Plots and saves a video of PSO training across epochs with reconstructions

    % Overwrite existing video file
    if exist(filename, 'file')
        delete(filename);
    end

    num_epochs = numel(particle_history);

    % Early exit if nothing to plot
    if all(cellfun(@isempty, particle_history)) || num_epochs == 0
        warning("No particle history to plot.");
        return;
    end

    particles = particle_history{find(~cellfun(@isempty, particle_history), 1)};  % Find first valid entry
    [num_particles, dim] = size(particles);
    downsampled_idx = round(linspace(1, dim, min(300, dim)));

    show_recon = nargin > 3 && ~isempty(X_hats);
    show_best  = nargin > 1 && ~isempty(g_best_history);

    v = VideoWriter(filename, 'MPEG-4');
    v.FrameRate = 10;
    open(v);

    % Create figure once
    f = figure('Name', 'Swarm Training Video', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 500]);

    for epoch = 1:num_epochs
        clf(f);  % Clear figure
        t = tiledlayout(f, 1, 3, 'Padding', 'compact');

        if isempty(particle_history{epoch})
            continue;
        end

        particles = particle_history{epoch};
        [num_particles, dim] = size(particles);
        cols = downsampled_idx(downsampled_idx <= dim);

        % --- 1: Particle positions ---
        nexttile(t, 1);
        hold on;
        h_particles = gobjects(min(num_particles, 20), 1);
        for i = 1:min(num_particles, 20)
            h_particles(i) = plot(cols, particles(i, cols), '-', 'LineWidth', 1);
        end

        if show_best && size(g_best_history, 1) >= epoch
            h_best = plot(cols, g_best_history(epoch, cols), 'k-', 'LineWidth', 2);
            legend([h_particles(1), h_best], {'Particles', 'Global Best'});
        end

        hold off;
        title(sprintf('Particle Positions - Epoch %d', epoch));
        xlabel('Parameter Index'); ylabel('Value');
        xlim([1, dim]); ylim padded; grid on; box on;

        % --- 2: Particle deltas ---
        nexttile(t, 2);
        if epoch > 1 && ~isempty(particle_history{epoch-1})
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

        % --- 3: Reconstruction ---
        nexttile(t, 3);
        if show_recon && numel(X_hats) >= epoch && ~isempty(X_hats{epoch})
            x_test = x_train_selected;
            x_hat  = X_hats{epoch};
            plot(x_test, 'b', 'LineWidth', 1.2); hold on;
            plot(x_hat, 'r--', 'LineWidth', 1.2);
            legend('Original', 'Reconstructed');
            title('Reconstruction (Best Particle)');
            xlabel('Index'); ylabel('Voltage');
            axis tight; grid on;
        else
            axis off;
            text(0.5, 0.5, 'No reconstruction available', 'HorizontalAlignment', 'center', 'FontSize', 12);
        end

        drawnow;
        set(gcf, 'Renderer', 'opengl');
        frame = getframe(f);
        writeVideo(v, frame);
    end

    close(v);
    close(f);
end
