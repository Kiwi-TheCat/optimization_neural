function plot_particle_swarm_video(particle_history, g_best_history, x_test_log_out, every_n_epochs, filename)
    % Plots and saves a video of PSO training across epochs

    if nargin < 4 || isempty(every_n_epochs)
        every_n_epochs = 5;
    end
    if nargin < 5 || isempty(filename)
        filename = 'swarm_training_video.mp4';
    end

    % Overwrite video if exists
    if exist(filename, 'file')
        delete(filename);
    end

    num_epochs = numel(particle_history);
    [num_particles, dim] = size(particle_history{1});
    downsampled_idx = round(linspace(1, dim, min(300, dim)));

    show_recon = nargin > 2 && ~isempty(x_test_log_out);
    show_best  = nargin > 1 && ~isempty(g_best_history) && size(g_best_history, 2) > 1;

    v = VideoWriter(filename, 'MPEG-4');
    v.FrameRate = 10;
    open(v);

    % Create figure once outside loop
    f = figure('Name', 'Swarm Training Video', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 500]);
    t = tiledlayout(f, 1, 3, 'Padding', 'compact');

    for epoch = 1:num_epochs


        % Clear all tiles before updating
        clf(f);
        t = tiledlayout(f, 1, 3, 'Padding', 'compact');

        % --- 1: Particle positions ---
        nexttile(t, 1);
        particles = particle_history{epoch};
        hold on;
        for i = 1:min(num_particles, 20)
            plot(downsampled_idx, particles(i, downsampled_idx), '-', 'LineWidth', 1);
        end
        if show_best
            plot(downsampled_idx, g_best_history(epoch, downsampled_idx), 'k-', 'LineWidth', 2);
            legend('Particles', 'Global Best');
        end
        hold off;
        title(sprintf('Particle Positions - Epoch %d', epoch));
        xlabel('Parameter Index'); ylabel('Value');
        xlim([1, dim]); ylim([-0.5, 0.5]); grid on; box on;

        % --- 2: Particle deltas ---
        nexttile(t, 2);
        if epoch > 1
            delta = particles - particle_history{epoch - 1};
            delta = delta(:, downsampled_idx);
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
        if show_recon && ~isempty(x_test_log_out{epoch})
            x_test = x_test_log_out{epoch}(1, :);
            x_hat  = x_test_log_out{epoch}(2, :);
            plot(x_test, 'b', 'LineWidth', 1.2); hold on;
            plot(x_hat, 'r--', 'LineWidth', 1.2);
            legend('Original', 'Reconstructed');
            title('Reconstruction (best particle)');
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
