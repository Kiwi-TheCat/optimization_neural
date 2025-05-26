function visualize_weight_static(weights_log, x_test_log)
    % VISUALIZE_WEIGHT_STATIC Visualizes encoder/decoder weights over training
    % Inputs:
    % - weights_log: struct containing weights over epochs
    % - x_test, x_hat: input and reconstructed signal (optional, same size)

    f_overlay = figure(101);
    set(f_overlay, 'Name', 'Weight Visualization (Static)', 'Position', [200, 200, 1800, 400]);

    for o = 1:numel(weights_log)
        optimizer_type = weights_log(o).optimizer;
        num_epochs = numel(weights_log(o).epoch);

        for epoch = 1:num_epochs
            tiledlayout(1,3, 'Padding', 'compact');
            
            x_hat = x_test_log{epoch, o}(2,:);
            x_test = x_test_log{epoch, o}(1,:);
            % (1) Plot Input vs Reconstruction
            nexttile;
                plot(x_test, 'b'); hold on;
                plot(x_hat, 'r'); hold off;
                legend('Original', 'Reconstructed');

            title(sprintf('Input vs Reconstruction - %s - Epoch %d', upper(optimizer_type), epoch));
            xlabel('Feature Index'); ylabel('Voltage'); axis tight;

            % (2) Decoder weights (log10 absolute)
            nexttile;
            w_dec = log10(abs(weights_log(o).epoch(epoch).Wd_output) + 1e-6);
            imagesc(w_dec); axis tight; caxis([-1, 1]); colorbar;
            colormap(gca, parula);
            title('Decoder Weights (log10)');
            xlabel('Output Features'); ylabel('Decoder Hidden'); set(gca,'YDir','normal');

            % (3) Encoder weights (log10 absolute)
            nexttile;
            w_enc = log10(abs(weights_log(o).epoch(epoch).We1) + 1e-6);
            imagesc(w_enc); axis tight; caxis([-1, 1]); colorbar;
            colormap(gca, parula);
            title('Encoder Weights (log10)');
            xlabel('Hidden Units'); ylabel('Input Features'); set(gca,'YDir','normal');

            drawnow;
            pause(0.01); % Ensure update renders smoothly
        end
    end
end
