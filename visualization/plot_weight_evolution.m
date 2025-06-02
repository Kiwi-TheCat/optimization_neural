function plot_weight_evolution(weight_log, field_name, o, update_indices)
    % PLOT_WEIGHT_EVOLUTION - Visualize cumulative delta of weights over epochs.
    % If update_indices is not provided, it auto-selects the weights that changed.
    %
    % Parameters:
    %   weight_log     - Struct array containing .epoch and weight fields
    %   field_name     - Name of the weight field to visualize (e.g., 'We1')
    %   o              - Optimizer index in weight_log
    %   update_indices - Optional vector of linear indices to plot

    % --- Setup
    num_epochs = numel(weight_log(o).epoch);
    epoch_numbers = zeros(1, num_epochs);
    first_W = weight_log(o).epoch(1).(field_name);
    all_weights = zeros(numel(first_W), num_epochs);

    % --- Collect all weights over epochs
    for k = 1:num_epochs
        W = weight_log(o).epoch(k).(field_name);
        all_weights(:, k) = W(:);
        epoch_numbers(k) = weight_log(o).epoch(k).epoch;
    end

    % --- Auto-detect changing weights if not provided
    if nargin < 4 || isempty(update_indices)
        changed = any(diff(all_weights, 1, 2) ~= 0, 2);
        update_indices = find(changed);
        origin = ' (auto-detected)';
    else
        origin = ' (from input)';
    end

    % --- Calculate delta traces
    % delta = difference between weights across epochs (per weight)
    delta_traces = diff(all_weights(update_indices, :), 1, 2);
    %update_traces = cumsum(abs(delta_traces), 2);  % summed absolute delta over time
    update_traces = cumsum(delta_traces, 2); % summed signed delta over time

    % --- Plot heatmap of weight evolution
    figure('Name', sprintf('Summed Delta Weight Evolution: %s', field_name));
    imagesc(epoch_numbers(2:end), 1:numel(update_indices), update_traces);
    colorbar;
    xlabel('Epoch');
    ylabel('Weight Index');
    optimizer_name = upper(weight_log(o).optimizer);
    title(sprintf('Summed Δ of %d weights in %s%s [%s]', ...
    numel(update_indices), field_name, origin, optimizer_name));
end
