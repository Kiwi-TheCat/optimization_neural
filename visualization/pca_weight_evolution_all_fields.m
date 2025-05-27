function pca_weight_evolution_all_fields(weights_log, fieldnames)
% PCA_WEIGHT_EVOLUTION_ALL_FIELDS
% Visualizes PCA projection of multiple weight matrices across all optimizers.
% Each point = weights at one epoch; color = weight type, marker = optimizer

    if nargin < 2
        fieldnames = {'We1', 'We_latent', 'Wd1', 'Wd_output'};
    end
    if ischar(fieldnames) || isstring(fieldnames)
        fieldnames = {fieldnames};  % ensure it's a cell array
    end

    num_optim = numel(weights_log);
    optim_names = arrayfun(@(w) upper(w.optimizer), weights_log, 'UniformOutput', false);

    marker_styles = {'o', 'x', '^', 's', 'd', 'v'};  % markers for optimizers
    cmap = lines(numel(fieldnames));                % colors for weight types

    % --- Collect data ---
    all_weights = [];
    all_labels = [];
    all_fields = {};
    all_optimizers = {};

    for o = 1:num_optim
        optimizer_name = weights_log(o).optimizer;

        for f = 1:numel(fieldnames)
            fname = fieldnames{f};
            epochs = weights_log(o).epoch;

            for e = 1:numel(epochs)
                if isfield(epochs(e), fname)
                    W = epochs(e).(fname);
                    all_weights(end+1, :) = W(:)';
                    all_labels(end+1) = epochs(e).epoch;
                    all_fields{end+1} = fname;
                    all_optimizers{end+1} = optimizer_name;
                end
            end
        end
    end

    % --- PCA ---
    [coeff, score, ~] = pca(zscore(all_weights));

    % --- Plot ---
    figure('Name', 'PCA of All Weights', 'Position', [200, 150, 800, 600]); hold on;

    legends = {};
    for f = 1:numel(fieldnames)
        fname = fieldnames{f};
        color = cmap(f,:);

        for o = 1:num_optim
            optim = weights_log(o).optimizer;
            idx = strcmp(all_fields, fname) & strcmp(all_optimizers, optim);
            if any(idx)
                marker = marker_styles{mod(o-1, numel(marker_styles)) + 1};
                scatter(score(idx,1), score(idx,2), 60, ...
                        'Marker', marker, ...
                        'MarkerFaceColor', color, ...
                        'MarkerEdgeColor', 'k', ...
                        'DisplayName', [upper(optim) ' - ' fname]);
            end
        end
    end

    legend('Location', 'eastoutside');
    xlabel('PC 1'); ylabel('PC 2');
    title('PCA of Weight Matrices Across Epochs and Optimizers');
    axis equal tight;
    grid on;
end
