function animate_latent_pca_all(weights_log)
% ANIMATE_LATENT_PCA_ALL
% Animates PCA of latent activations across all epochs and optimizers.
%
% Input:
%   weights_log(o).epoch(e).z = [n_samples x latent_dim] latent activations

    num_optim = numel(weights_log);
    num_epochs = max(cellfun(@(s) s.epoch, {weights_log.epoch}));

    colors = parula(num_epochs);
    f = figure('Name', 'Latent Space PCA Evolution', ...
               'Position', [100, 100, 300 * num_optim, 400]);

    % Precompute PCA bases and collect score ranges
    coeffs = cell(num_optim, 1);
    all_scores = cell(num_optim, num_epochs);
    xlims = []; ylims = [];

    for o = 1:num_optim
        optimizer_epochs = weights_log(o).epoch;

        % Get PCA basis from first valid z
        for s = optimizer_epochs
            if isfield(s, 'z') && size(s.z,1) > 1 && all(std(s.z, 0, 1) > 0)
                [coeffs{o}, ~] = pca(zscore(s.z));
                break;
            end
        end

        % Project each z and track limits
        for i = 1:numel(optimizer_epochs)
            if ~isfield(optimizer_epochs(i), 'z'), continue; end
            z = optimizer_epochs(i).z;
            if isempty(z) || size(z, 1) < 2, continue; end
            z = zscore(z);

            if isempty(coeffs{o})
                [coeffs{o}, ~] = pca(z);
            end

            score = z * coeffs{o}(:,1:2);
            all_scores{o, i} = score;

            if ~isempty(score) && size(score,2) >= 2
                xlims = [xlims; min(score(:,1)), max(score(:,1))];
                ylims = [ylims; min(score(:,2)), max(score(:,2))];
            end
        end
    end

    % Safe axis limits
    if isempty(xlims) || isempty(ylims)
        warning('No valid latent activations to animate.');
        return;
    end

    x_range = [min(xlims(:,1)), max(xlims(:,2))];
    y_range = [min(ylims(:,1)), max(ylims(:,2))];

    % Animate epochs
    for e = 1:num_epochs
        clf(f);
        for o = 1:num_optim
            subplot(1, num_optim, o);
            optimizer = upper(weights_log(o).optimizer);
            title_text = sprintf('%s - Epoch %d', optimizer, e);

            if size(all_scores, 2) < e || isempty(all_scores{o, e})
                title([title_text, ' (no z)']);
                continue;
            end

            score = all_scores{o, e};
            scatter(score(:,1), score(:,2), 20, colors(e,:), 'filled');
            xlabel('PC 1'); ylabel('PC 2');
            title(title_text);
            xlim(x_range); ylim(y_range);
            axis equal tight;
        end

        sgtitle('Latent PCA Evolution Across Epochs');
        drawnow;
        pause(0.1);
    end
end
