function visualize_weight_animation(weights_log, x_test_log)
    % VISUALIZE_WEIGHT_ANIMATION Animates decoder and encoder weight changes across epochs

    % Create figure for animation
    f_overlay = figure(100);
    set(f_overlay, 'Name', 'Training Visualization', 'Position', [200, 200, 1800, 400]);

    % Loop through each optimizer's training log
    for o = 1:numel(weights_log)
        optimizer_type = weights_log(o).optimizer;
        num_epochs = numel(weights_log(o).epoch);
        
        for epoch = 1:num_epochs
            % Dummy input and reconstruction for visualization
            %x_test = rand(1, size(weights_log(o).epoch(epoch).We1, 1));
            %x_hat = rand(1, size(weights_log(o).epoch(epoch).We1, 1));
            x_hat = x_test_log{epoch, o}(2,:);
            x_test = x_test_log{epoch, o}(1,:);
            % Plot Input vs Reconstruction
            subplot(1,3,1);
            plot(x_test, 'b'); hold on;
            plot(x_hat, 'r'); hold off;
            title(sprintf('Input vs Reconstruction - %s - Epoch %d', upper(optimizer_type), epoch));
            legend('Original', 'Reconstructed');
            xlabel('Feature Index'); ylabel('Voltage'); axis tight;

            % Plot Δ Decoder Weights
            subplot(1,3,2);
            if epoch > 1
                delta_decoder = log10(abs(weights_log(o).epoch(epoch).Wd_output) + 1e-6) - ...
                                 log10(abs(weights_log(o).epoch(epoch-1).Wd_output) + 1e-6);
            else
                delta_decoder = zeros(size(weights_log(o).epoch(epoch).Wd_output));
            end
            imagesc(delta_decoder); axis tight;
            caxis([-0.2, 0.2]);
            colorbar;
            n = 128;
            blue_to_black = [linspace(0,0,n)', linspace(0,0,n)', linspace(1,0,n)'];
            black_to_green = [linspace(0,0,n)', linspace(0,1,n)', linspace(0,0,n)'];
            custom_cmap = [blue_to_black; 0 0 0; black_to_green];
            colormap(gca, custom_cmap);
            title(sprintf('\x0394 Decoder Weights - %s - Epoch %d', upper(optimizer_type), epoch));
            xlabel('Output Features'); ylabel('Decoder Hidden'); set(gca,'YDir','normal');

            % Plot Δ Encoder Weights
            subplot(1,3,3);
            if epoch > 1
                delta_encoder = log10(abs(weights_log(o).epoch(epoch).We1) + 1e-6) - ...
                                 log10(abs(weights_log(o).epoch(epoch-1).We1) + 1e-6);
            else
                delta_encoder = zeros(size(weights_log(o).epoch(epoch).We1));
            end
            imagesc(delta_encoder); axis tight;
            caxis([-1, 1]);
            colorbar;
            colormap(gca, custom_cmap);
            title(sprintf('\x0394 Encoder Weights - %s - Epoch %d', upper(optimizer_type), epoch));
            xlabel('Hidden Units'); ylabel('Input Features'); set(gca,'YDir','normal');

            drawnow;
        end
    end
end
