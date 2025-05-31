function plot_particle_swarm(particle_history, g_best_history, x_test_log_out, every_n_epochs)
    % Visualizes swarm evolution, delta, and signal reconstruction during PSO training

    if nargin < 4
        every_n_epochs = 5;
    end

    num_epochs = numel(particle_history);
    [num_particles, dim] = size(particle_history{1});
    downsampled_idx = round(linspace(1, dim, min(300, dim)));

    show_recon = nargin > 2 && ~isempty(x_test_log_out);
    show_best  = nargin > 1 && ~isempty(g_best_history) && size(g_best_history, 2) > 1;

    % Prepare figure
    f = figure('Name', 'Swarm Training Monitor', ...
               'NumberTitle', 'off', 'Position', [100, 100, 1600, 500]);
    tiledlayout(1,3,'Padding','compact');

    for epoch = 1:num_epochs
        if ~mod(epoch, every_n_epochs) && epoch < num_epochs
            continue;
        end

        if ~isvalid(f), return; end
        nexttile(1); cla;
        particles = particle_history{epoch};

        % Plot all particles (downsampled)
        hold on;
        for i = 1:min(num_particles, 20)
            plot(downsampled_idx, particles(i, downsampled_idx), '-', 'LineWidth', 1);
        end
        if show_best
            plot(downsampled_idx, g_best_history(epoch, downsampled_idx), 'k-', 'LineWidth', 2);
            legend('Particles', 'Global Best');
        end
        hold off;
        title(sprintf('Particle Positions – Epoch %d', epoch));
        xlabel('Parameter Index'); ylabel('Value');
        xlim([1, dim]); ylim([-1.5, 1.5]); grid on; box on;

        % Delta subplot
        nexttile(2); cla;
        if epoch > 1
            delta = particles - particle_history{epoch - 1};
            delta = delta(:, downsampled_idx);
            imagesc(delta);
            caxis([-0.05, 0.05]);
            colorbar;
            xlabel('Parameter Index'); ylabel('Particle #');
            title(sprintf('\\Delta Weights – Epoch %d', epoch));
            set(gca, 'YDir', 'normal');
        else
            axis off;
            text(0.5, 0.5, 'Initial epoch (no delta)', ...
                'HorizontalAlignment', 'center', 'FontSize', 12);
        end

        % Reconstruction subplot
        nexttile(3); cla;
        if show_recon && ~isempty(x_test_log_out{epoch})
            x_test = x_test_log_out{epoch}(1, :);
            x_hat  = x_test_log_out{epoch}(2, :);
            plot(x_test, 'b', 'LineWidth', 1.2); hold on;
            plot(x_hat, 'r--', 'LineWidth', 1.2);
            legend('Original', 'Reconstructed');
            title('Reconstruction (first signal)');
            xlabel('Index'); ylabel('Voltage');
            axis tight; grid on;
        else
            axis off;
            text(0.5, 0.5, 'No reconstruction available', ...
                'HorizontalAlignment', 'center', 'FontSize', 12);
        end

        drawnow;
        pause(0.05);
    end
end
