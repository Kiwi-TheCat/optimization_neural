function plot_weight_evolution(weight_log, field_name, update_indices)
    num_updates = length(update_indices);
    % Collect selected weights over epochs
    num_epochs = length(weight_log.epoch);
    update_traces = zeros(num_updates, num_epochs);

    for k = 1:num_epochs
        W = weight_log.epoch(k).(field_name);
        W_flat = W(:);  % flatten
        update_traces(:, k) = W_flat(update_indices);
    end
    clim = [min(update_traces(:)), max(update_traces(:))];
    % Plot evolution of selected weights
    figure;
    imagesc(update_traces);
    caxis(clim_enc); colorbar;
    xlabel('Epoch');
    ylabel('Selected Weight Index');
    title(sprintf('Evolution of %d random weights in %s', num_updates, field_name));
end
