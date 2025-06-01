function visualize_weight_animation(weights_log, x_test_log, update_indices)
    % VISUALIZE_WEIGHT_ANIMATION - Animates weight values at given indices across epochs.
    % If update_indices is not given, selects up to 200 changed weights automatically.

    f_overlay = figure(100);
    set(f_overlay, 'Name', 'Training Visualization', 'Position', [200, 200, 1800, 400]);

    for o = 1:numel(weights_log)
        optimizer_type = weights_log(o).optimizer;
        num_epochs = numel(weights_log(o).epoch);

        % === Determine update_indices if not provided ===
        if nargin < 3 || isempty(update_indices)
            origin = 'from changed weights';

            We_ref = weights_log(o).epoch(1).We1(:);
            changes = false(size(We_ref));

            for epoch = 2:num_epochs
                We_curr = weights_log(o).epoch(epoch).We1(:);
                changes = changes | (We_curr ~= We_ref);
                We_ref = We_curr;
            end

            update_indices = find(changes);
            if numel(update_indices) > 200
                rng(42);  % for reproducibility
                update_indices = update_indices(randperm(numel(update_indices), 200));
            end
        else
            origin = 'from given indices';
        end

        % === Collect weight traces over epochs ===
        encoder_traces = zeros(length(update_indices), num_epochs);
        decoder_traces = zeros(length(update_indices), num_epochs);
        
        for epoch = 1:num_epochs
            We_flat = weights_log(o).epoch(epoch).We1(:);
            Wd_flat = weights_log(o).epoch(epoch).Wd_output(:);
            encoder_traces(:, epoch) = We_flat(update_indices);
            decoder_traces(:, epoch) = Wd_flat(update_indices);
        end
        % Set color limits based on full trace range
        clim_enc = [min(encoder_traces(:)), max(encoder_traces(:))];
        clim_dec = [min(decoder_traces(:)), max(decoder_traces(:))];

        % === Animate ===
        for epoch = 1:num_epochs
            x_hat = x_test_log{epoch, o}(2,:);
            x_test = x_test_log{epoch, o}(1,:);

            subplot(1, 3, 1);
            plot(x_test, 'b'); hold on;
            plot(x_hat, 'r'); hold off;
            title(sprintf('Input vs Reconstruction - %s - Epoch %d', upper(optimizer_type), 10*epoch));
            legend('Original', 'Reconstructed');
            xlabel('Feature Index'); ylabel('Voltage'); axis tight;

            subplot(1, 3, 2);
            imagesc(decoder_traces(:, 1:epoch));  % auto-scaled
            caxis(clim_dec); colorbar;
            ylabel('Decoder Weight Index'); xlabel('Epoch');
            title(sprintf('Decoder Weights (%s) - %s - Epoch %d', origin, upper(optimizer_type), 10*epoch));

            subplot(1, 3, 3);
            imagesc(encoder_traces(:, 1:epoch));  % auto-scaled
            caxis(clim_enc); colorbar;
            ylabel('Encoder Weight Index'); xlabel('Epoch');
            title(sprintf('Encoder Weights (%s) - %s - Epoch %d', origin, upper(optimizer_type), 10*epoch));

            drawnow;
        end
    end
end
