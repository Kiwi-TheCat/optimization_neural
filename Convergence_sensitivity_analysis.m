%% Convergence Sensitivity Analysis - Using Pre-trained Data
% Load and analyze convergence from previously trained noisy weights

clear; clc;

% Load the training results from noise analysis
try
    load('enhanced_trained_noisy_weights.mat'); % This contain the training results
    fprintf('Loaded training results from enhanced_trained_noisy_weights.mat\n');
catch
    fprintf('Error: trained_noisy_weights.mat not found!\n');
    fprintf('Please run your noise training analysis first and save results as:\n');
    fprintf('save(''trained_noisy_weights.mat'', ''results'', ''loss_histories'', ''original_params'', ''final_params_all'');\n');
    return;
end

% Also load original data for validation
load("Adam_weights_and_bias.mat");
load("data_1s.mat");

% Preprocess data (same as training)
data = data(1:end-1,1:end-1);
delta = diff(data);
X_train_original = delta(1:384, :);
X_train_original = double(X_train_original);
mean_X = mean(X_train_original, 1);
std_X = std(X_train_original, 0, 1);
std_X(std_X == 0) = 1e-6;
X_train = (X_train_original - mean_X) ./ std_X;
relu = @(x) max(0, x);

fprintf('=== CONVERGENCE SENSITIVITY ANALYSIS ===\n');

%% 1. Extract and organize the training data
% Assume results table has: LayerName, NoiseLevel, InitialLoss, FinalLoss
% Assume loss_histories contains the epoch-by-epoch loss for each experiment

% Get unique combinations
unique_layers = unique(results.LayerName);
unique_noise_levels = unique(results.NoiseLevel);
unique_noise_levels = unique_noise_levels(unique_noise_levels > 0); % Exclude original (0 noise)

fprintf('Found data for %d layers and %d noise levels\n', length(unique_layers), length(unique_noise_levels));

%% 2. Convergence Speed Analysis
fprintf('\n=== CONVERGENCE SPEED ANALYSIS ===\n');

convergence_summary = table('Size', [0 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'logical'}, ...
    'VariableNames', {'LayerName', 'NoiseLevel', 'ConvergenceEpoch', 'ConvergenceSpeed', 'FinalLoss', 'ReachedGlobalMin'});

% Get baseline (original) performance
original_data = results(results.NoiseLevel == 0, :);
if isempty(original_data)
    % Calculate original loss if not in results
    original_loss = evaluate_loss(params, X_train, relu);
    fprintf('Baseline loss: %.6f\n', original_loss);
else
    original_loss = original_data.FinalLoss(1);
    fprintf('Baseline loss: %.6f\n', original_loss);
end

% Analyze each experiment
experiment_idx = 1;
for i = 1:height(results)
    if results.NoiseLevel(i) == 0
        continue; % Skip original
    end
    
    layer_name = results.LayerName{i};
    noise_level = results.NoiseLevel(i);
    initial_loss = results.InitialLoss(i);
    final_loss = results.FinalLoss(i);
    
    % Get loss history for this experiment
    if exist('loss_histories', 'var') && length(loss_histories) >= experiment_idx
        loss_history = loss_histories{experiment_idx};
        
        % Calculate convergence metrics
        convergence_epoch = find_convergence_epoch(loss_history);
        convergence_speed = calculate_convergence_speed(loss_history);
        
        fprintf('Experiment %d (%s, noise %.3f):\n', experiment_idx, layer_name, noise_level);
        fprintf('  Converged at epoch %d, 95%% improvement by epoch %d\n', convergence_epoch, convergence_speed);
    else
        % If no loss history, estimate from initial/final loss
        convergence_epoch = NaN;
        convergence_speed = NaN;
        fprintf('Experiment %d (%s, noise %.3f): No loss history available\n', experiment_idx, layer_name, noise_level);
    end
    
    % Check if reached global minimum (within 1% tolerance)
    reached_global_min = abs(final_loss - original_loss) < 0.05 * original_loss;
    
    % Recovery analysis
    recovery_percent = (initial_loss - final_loss) / initial_loss * 100;
    performance_vs_original = (final_loss - original_loss) / original_loss * 100;
    
    fprintf('  Recovery: %.1f%%, Performance vs original: %+.1f%%\n', recovery_percent, performance_vs_original);
    fprintf('  Reached global minimum: %s\n', mat2str(reached_global_min));
    
    % Store in summary table
    convergence_summary = [convergence_summary; {layer_name, noise_level, convergence_epoch, convergence_speed, final_loss, reached_global_min}];
    
    experiment_idx = experiment_idx + 1;
end

%% 3. Visualization
figure('Position', [100, 100, 1400, 1000]);

% Plot 1: Loss histories (if available)
if exist('loss_histories', 'var')
    subplot(2, 3, 1);
    hold on;
    
    experiment_idx = 1;
    colors = lines(length(unique_layers));
    layer_color_map = containers.Map(unique_layers, num2cell(colors, 2));
    
    for i = 1:height(results)
        if results.NoiseLevel(i) == 0 || experiment_idx > length(loss_histories)
            continue;
        end
        
        if ~isempty(loss_histories{experiment_idx})
            layer_name = results.LayerName{i};
            noise_level = results.NoiseLevel(i);
            loss_history = loss_histories{experiment_idx};
            
            color = layer_color_map(layer_name);
            line_style = get_line_style(noise_level);
            
            plot(1:length(loss_history), loss_history, 'Color', color, 'LineStyle', line_style, ...
                'LineWidth', 1.5, 'DisplayName', sprintf('%s (%.3f)', layer_name, noise_level));
        end
        experiment_idx = experiment_idx + 1;
    end
    
    xlabel('Epoch');
    ylabel('Loss');
    title('Convergence Curves');
    legend('Location', 'best');
    grid on;
    hold off;
end

% Plot 2: Convergence Speed vs Noise Level
subplot(2, 3, 2);
valid_data = convergence_summary(~isnan(convergence_summary.ConvergenceSpeed), :);
if ~isempty(valid_data)
    for layer = unique_layers'
        layer_data = valid_data(strcmp(valid_data.LayerName, layer), :);
        if ~isempty(layer_data)
            plot(layer_data.NoiseLevel, layer_data.ConvergenceSpeed, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', layer{1});
            hold on;
        end
    end
    xlabel('Noise Level');
    ylabel('Convergence Speed (epochs)');
    title('Convergence Speed vs Noise');
    legend('Location', 'best');
    grid on;
    set(gca, 'XScale', 'log');
end

% Plot 3: Global Minimum Convergence Rate
subplot(2, 3, 3);
global_min_rates = zeros(length(unique_layers), length(unique_noise_levels));
for i = 1:length(unique_layers)
    for j = 1:length(unique_noise_levels)
        layer_noise_data = convergence_summary(strcmp(convergence_summary.LayerName, unique_layers{i}) & ...
                                              convergence_summary.NoiseLevel == unique_noise_levels(j), :);
        if ~isempty(layer_noise_data)
            global_min_rates(i, j) = sum(layer_noise_data.ReachedGlobalMin) / height(layer_noise_data) * 100;
        end
    end
end

bar_data = global_min_rates';
bar(bar_data);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.3f', x), unique_noise_levels, 'UniformOutput', false));
xlabel('Noise Level');
ylabel('Global Minimum Rate (%)');
title('Global Minimum Convergence Rate');
legend(unique_layers, 'Location', 'best');
grid on;

% Plot 4: Final Loss Distribution
subplot(2, 3, 4);
hold on;
yline(original_loss, 'k--', 'LineWidth', 2, 'DisplayName', 'Original');
for i = 1:length(unique_layers)
    layer_data = convergence_summary(strcmp(convergence_summary.LayerName, unique_layers{i}), :);
    scatter(layer_data.NoiseLevel, layer_data.FinalLoss, 100, 'filled', 'DisplayName', unique_layers{i});
end
xlabel('Noise Level');
ylabel('Final Loss');
title('Final Loss vs Noise Level');
legend('Location', 'best');
grid on;
set(gca, 'XScale', 'log');

% Plot 5: Weight Space Analysis (if final parameters available)
if exist('final_params_all', 'var')
    subplot(2, 3, 5);
    analyze_weight_space_distances(params, final_params_all, results);
end

% Plot 6: Summary Statistics
subplot(2, 3, 6);
% Create summary bar chart
summary_stats = zeros(length(unique_layers), 3); % [avg_recovery, avg_speed, global_min_rate]
for i = 1:length(unique_layers)
    layer_data = convergence_summary(strcmp(convergence_summary.LayerName, unique_layers{i}), :);
    if ~isempty(layer_data)
        layer_results = results(strcmp(results.LayerName, unique_layers{i}) & results.NoiseLevel > 0, :);
        if ~isempty(layer_results)
            avg_recovery = mean((layer_results.InitialLoss - layer_results.FinalLoss) ./ layer_results.InitialLoss * 100);
            avg_speed = mean(layer_data.ConvergenceSpeed, 'omitnan');
            global_min_rate = sum(layer_data.ReachedGlobalMin) / height(layer_data) * 100;
            
            summary_stats(i, :) = [avg_recovery, avg_speed/10, global_min_rate]; % Scale speed for visualization
        end
    end
end

bar(summary_stats);
set(gca, 'XTickLabel', unique_layers);
ylabel('Percentage / Scaled Value');
title('Summary Statistics by Layer');
legend({'Avg Recovery (%)', 'Avg Speed (/10)', 'Global Min Rate (%)'}, 'Location', 'best');
grid on;

sgtitle('Convergence Sensitivity Analysis Results');

%% 4. Summary Report
fprintf('\n=== SUMMARY REPORT ===\n');
fprintf('Analysis of %d experiments across %d layers and %d noise levels\n', ...
    height(convergence_summary), length(unique_layers), length(unique_noise_levels));

for layer = unique_layers'
    layer_data = convergence_summary(strcmp(convergence_summary.LayerName, layer), :);
    if ~isempty(layer_data)
        fprintf('\n%s Layer Analysis:\n', layer{1});
        fprintf('  Global minimum rate: %.0f%% (reached original performance)\n', ...
            sum(layer_data.ReachedGlobalMin) / height(layer_data) * 100);
        fprintf('  Average convergence speed: %.1f epochs\n', mean(layer_data.ConvergenceSpeed, 'omitnan'));
        fprintf('  Noise sensitivity: %s\n', assess_noise_sensitivity(layer_data));
    end
end

%% Enhanced Convergence Analysis - Adding Missing Metrics
% Add this to your existing convergence analysis code

%% Additional Analysis - Same Minimum Rate with Multiple Thresholds
fprintf('\n=== SAME MINIMUM RATE ANALYSIS ===\n');

% Test different tolerance levels
tolerance_levels = [0.01, 0.05, 0.10]; % 1%, 5%, 10%
tolerance_names = {'Strict (1%)', 'Medium (5%)', 'Relaxed (10%)'};

for t = 1:length(tolerance_levels)
    fprintf('\n--- %s Tolerance ---\n', tolerance_names{t});
    
    for noise_level = unique_noise_levels'
        noise_data = convergence_summary(convergence_summary.NoiseLevel == noise_level, :);
        
        if ~isempty(noise_data)
            % Calculate same minimum rate for this tolerance
            tolerance = tolerance_levels(t);
            same_min_count = sum(abs(noise_data.FinalLoss - original_loss) < tolerance * original_loss);
            same_min_rate = same_min_count / height(noise_data) * 100;
            
            fprintf('Noise %.3f: Same minimum rate = %.0f%%\n', noise_level, same_min_rate);
        end
    end
end

%% Enhanced Weight Space Analysis
fprintf('\n=== ENHANCED WEIGHT SPACE ANALYSIS ===\n');

if exist('final_params_all', 'var') && ~isempty(final_params_all)
    
    % Extract original weights vector
    original_weights_vector = extract_weights_vector(original_params);
    
    % Initialize storage for detailed weight analysis
    weight_analysis = table('Size', [0 7], ...
        'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'logical'}, ...
        'VariableNames', {'LayerName', 'NoiseLevel', 'EuclideanDistance', 'RelativeDistance', 'CosineSimilarity', 'FinalLoss', 'SameDirection'});
    
    experiment_idx = 1;
    for i = 1:height(results)
        if results.NoiseLevel(i) == 0 || experiment_idx > length(final_params_all)
            continue;
        end
        
        if ~isempty(final_params_all{experiment_idx})
            layer_name = results.LayerName{i};
            noise_level = results.NoiseLevel(i);
            final_loss = results.FinalLoss(i);
            
            % Extract final weights
            final_weights_vector = extract_weights_vector(final_params_all{experiment_idx});
            
            if ~isempty(final_weights_vector) && length(final_weights_vector) == length(original_weights_vector)
                % Calculate weight space metrics
                euclidean_dist = norm(final_weights_vector - original_weights_vector);
                relative_dist = euclidean_dist / norm(original_weights_vector);
                
                % Cosine similarity
                dot_product = dot(final_weights_vector, original_weights_vector);
                cos_sim = dot_product / (norm(final_weights_vector) * norm(original_weights_vector));
                
                % Same direction check (cosine similarity > 0.95)
                same_direction = cos_sim > 0.95;
                
                % Store results
                weight_analysis = [weight_analysis; {layer_name, noise_level, euclidean_dist, relative_dist, cos_sim, final_loss, same_direction}];
                
                fprintf('Layer %s, Noise %.3f:\n', layer_name, noise_level);
                fprintf('  Euclidean Distance: %.4f\n', euclidean_dist);
                fprintf('  Relative Distance: %.4f\n', relative_dist);
                fprintf('  Cosine Similarity: %.6f\n', cos_sim);
                fprintf('  Same Direction: %s\n', mat2str(same_direction));
                
                % Interpret results
                if relative_dist < 0.1
                    fprintf('  → Same global minimum (low relative distance)\n');
                elseif cos_sim > 0.95
                    fprintf('  → Same solution direction\n');
                elseif euclidean_dist > 1000 && relative_dist < 0.5
                    fprintf('  → Potentially scaled solution\n');
                else
                    fprintf('  → Different local minimum\n');
                end
                fprintf('\n');
            end
        end
        experiment_idx = experiment_idx + 1;
    end
    
    % Summary statistics for weight space analysis
    fprintf('=== WEIGHT SPACE SUMMARY ===\n');
    fprintf('Average relative distance: %.4f\n', mean(weight_analysis.RelativeDistance));
    fprintf('Average cosine similarity: %.6f\n', mean(weight_analysis.CosineSimilarity));
    fprintf('Same direction rate: %.0f%%\n', sum(weight_analysis.SameDirection) / height(weight_analysis) * 100);
    
else
    fprintf('No final parameters available for weight space analysis\n');
end

%% Convergence Speed Analysis Enhancement
fprintf('\n=== CONVERGENCE SPEED DETAILED ANALYSIS ===\n');

if exist('loss_histories', 'var') && ~isempty(loss_histories)
    
    % Get baseline convergence speed (if available)
    baseline_speed = NaN; % You'd need to save this from original training
    
    speed_analysis = table('Size', [0 6], ...
        'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'string'}, ...
        'VariableNames', {'LayerName', 'NoiseLevel', 'ConvergenceSpeed', 'SpeedVsBaseline', 'FinalLoss', 'Interpretation'});
    
    experiment_idx = 1;
    for i = 1:height(results)
        if results.NoiseLevel(i) == 0 || experiment_idx > length(loss_histories)
            continue;
        end
        
        if ~isempty(loss_histories{experiment_idx})
            layer_name = results.LayerName{i};
            noise_level = results.NoiseLevel(i);
            final_loss = results.FinalLoss(i);
            loss_history = loss_histories{experiment_idx};
            
            % Calculate convergence speed
            conv_speed = calculate_convergence_speed(loss_history);
            
            % Compare to baseline (if available)
            if ~isnan(baseline_speed)
                speed_ratio = conv_speed / baseline_speed;
                if speed_ratio < 0.9
                    interpretation = 'Faster - noise helps escape shallow minima';
                elseif speed_ratio > 1.1
                    interpretation = 'Slower - noise makes optimization harder';
                else
                    interpretation = 'Similar - robust optimization landscape';
                end
            else
                speed_ratio = NaN;
                interpretation = 'No baseline for comparison';
            end
            
            % Store results
            speed_analysis = [speed_analysis; {layer_name, noise_level, conv_speed, speed_ratio, final_loss, interpretation}];
        end
        experiment_idx = experiment_idx + 1;
    end
    
    % Display speed analysis
    fprintf('Convergence Speed Analysis:\n');
    disp(speed_analysis);
    
    % Speed vs noise level analysis
    for layer = unique_layers'
        layer_data = speed_analysis(strcmp(speed_analysis.LayerName, layer), :);
        if height(layer_data) > 1
            fprintf('\n%s Layer Speed Trend:\n', layer{1});
            [sorted_noise, idx] = sort(layer_data.NoiseLevel);
            sorted_speeds = layer_data.ConvergenceSpeed(idx);
            
            if sorted_speeds(end) > sorted_speeds(1) * 1.2
                fprintf('  Speed decreases with noise (%.1f → %.1f epochs)\n', sorted_speeds(1), sorted_speeds(end));
                fprintf('  → Noise makes optimization harder\n');
            elseif sorted_speeds(end) < sorted_speeds(1) * 0.8
                fprintf('  Speed increases with noise (%.1f → %.1f epochs)\n', sorted_speeds(1), sorted_speeds(end));
                fprintf('  → Noise helps escape shallow local minima\n');
            else
                fprintf('  Speed relatively stable (%.1f ± %.1f epochs)\n', mean(sorted_speeds), std(sorted_speeds));
                fprintf('  → Robust optimization landscape\n');
            end
        end
    end
end

%% Enhanced Visualization
figure('Position', [100, 100, 1600, 1200]);

% Plot 1: Same Minimum Rate with Multiple Thresholds (TOP LEFT)
subplot(2, 2, 1);
if exist('tolerance_levels', 'var')
    for t = 1:length(tolerance_levels)
        same_min_rates = zeros(1, length(unique_noise_levels));
        for j = 1:length(unique_noise_levels)
            noise_level = unique_noise_levels(j);
            noise_data = convergence_summary(convergence_summary.NoiseLevel == noise_level, :);
            if ~isempty(noise_data)
                tolerance = tolerance_levels(t);
                same_min_count = sum(abs(noise_data.FinalLoss - original_loss) < tolerance * original_loss);
                same_min_rates(j) = same_min_count / height(noise_data) * 100;
            end
        end
        semilogx(unique_noise_levels, same_min_rates, '-o', 'LineWidth', 2, 'DisplayName', tolerance_names{t});
        hold on;
    end
    xlabel('Noise Level');
    ylabel('Same Minimum Rate (%)');
    title('Same Minimum Rate vs Tolerance');
    legend('Location', 'best');
    grid on;
end

% Plot 2: Cosine Similarity Analysis (TOP RIGHT)
if exist('weight_analysis', 'var') && ~isempty(weight_analysis)
    subplot(2, 2, 2);
    scatter(weight_analysis.NoiseLevel, weight_analysis.CosineSimilarity, 100, 'filled');
    set(gca, 'XScale', 'log');
    xlabel('Noise Level');
    ylabel('Cosine Similarity');
    title('Weight Direction Similarity');
    yline(0.95, 'r--', 'Same Direction Threshold');
    grid on;
    
    % Plot 3: Relative vs Euclidean Distance (BOTTOM LEFT)
    subplot(2, 2, 3);
    scatter(weight_analysis.EuclideanDistance, weight_analysis.RelativeDistance, 100, weight_analysis.NoiseLevel, 'filled');
    xlabel('Euclidean Distance');
    ylabel('Relative Distance');
    title('Weight Space Distance Analysis');
    colorbar;
    colormap(gca, 'parula');
    yline(0.1, 'r--', 'Same Minimum Threshold');
    grid on;
end

% Plot 4: Loss Evolution by Layer (BOTTOM RIGHT)
if exist('weight_analysis', 'var') && ~isempty(weight_analysis)
    subplot(2, 2, 4);
    for layer = unique_layers'
        layer_data = weight_analysis(strcmp(weight_analysis.LayerName, layer), :);
        if ~isempty(layer_data)
            semilogx(layer_data.NoiseLevel, layer_data.FinalLoss, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', layer{1});
            hold on;
        end
    end
    yline(original_loss, 'k--', 'LineWidth', 2, 'DisplayName', 'Original');
    xlabel('Noise Level');
    ylabel('Final Loss');
    title('Final Loss Evolution by Layer');
    legend('Location', 'best');
    grid on;
end

sgtitle('Enhanced Convergence Sensitivity Analysis');

%% Summary Report Enhancement
fprintf('\n=== ENHANCED SUMMARY REPORT ===\n');

% A. Same Minimum Rate Analysis
fprintf('\nA. SAME MINIMUM RATE:\n');
for noise_level = unique_noise_levels'
    noise_data = convergence_summary(convergence_summary.NoiseLevel == noise_level, :);
    if ~isempty(noise_data)
        strict_rate = sum(abs(noise_data.FinalLoss - original_loss) < 0.01 * original_loss) / height(noise_data) * 100;
        medium_rate = sum(abs(noise_data.FinalLoss - original_loss) < 0.05 * original_loss) / height(noise_data) * 100;
        relaxed_rate = sum(abs(noise_data.FinalLoss - original_loss) < 0.10 * original_loss) / height(noise_data) * 100;
        
        fprintf('Noise %.3f: Strict=%0.f%%, Medium=%0.f%%, Relaxed=%0.f%%\n', ...
            noise_level, strict_rate, medium_rate, relaxed_rate);
    end
end

% B. Convergence Speed Analysis
fprintf('\nB. CONVERGENCE SPEED:\n');
if exist('speed_analysis', 'var') && ~isempty(speed_analysis)
    avg_speed = mean(speed_analysis.ConvergenceSpeed);
    speed_std = std(speed_analysis.ConvergenceSpeed);
    
    if speed_std / avg_speed < 0.1
        fprintf('Similar speed across conditions (%.1f ± %.1f epochs)\n', avg_speed, speed_std);
        fprintf('→ Robust optimization landscape\n');
    else
        fprintf('Variable speed across conditions (%.1f ± %.1f epochs)\n', avg_speed, speed_std);
        fprintf('→ Noise affects optimization difficulty\n');
    end
end

% C. Weight Space Analysis
fprintf('\nC. WEIGHT SPACE ANALYSIS:\n');
if exist('weight_analysis', 'var') && ~isempty(weight_analysis)
    avg_rel_dist = mean(weight_analysis.RelativeDistance);
    avg_cos_sim = mean(weight_analysis.CosineSimilarity);
    same_dir_rate = sum(weight_analysis.SameDirection) / height(weight_analysis) * 100;
    
    fprintf('Average relative distance: %.3f\n', avg_rel_dist);
    fprintf('Average cosine similarity: %.3f\n', avg_cos_sim);
    fprintf('Same direction rate: %.0f%%\n', same_dir_rate);
    
    if avg_rel_dist < 0.1
        fprintf('→ Same global minimum\n');
    elseif avg_cos_sim > 0.95
        fprintf('→ Same solution direction\n');
    else
        fprintf('→ Different local minima\n');
    end
end


%% Supporting Functions

function convergence_epoch = find_convergence_epoch(loss_history)
    % Find when loss stops decreasing significantly
    if length(loss_history) < 10
        convergence_epoch = length(loss_history);
        return;
    end
    
    window_size = 10;
    threshold = 1e-6;
    
    for i = window_size:length(loss_history)
        recent_improvement = loss_history(i-window_size+1) - loss_history(i);
        if recent_improvement < threshold
            convergence_epoch = i;
            return;
        end
    end
    convergence_epoch = length(loss_history);
end

function speed = calculate_convergence_speed(loss_history)
    % Find epoch where 95% of total improvement is achieved
    if length(loss_history) < 2
        speed = 1;
        return;
    end
    
    initial_loss = loss_history(1);
    final_loss = loss_history(end);
    
    if initial_loss <= final_loss
        speed = 1; % No improvement
        return;
    end
    
    total_improvement = initial_loss - final_loss;
    target_loss = initial_loss - 0.95 * total_improvement;
    
    speed = find(loss_history <= target_loss, 1);
    if isempty(speed)
        speed = length(loss_history);
    end
end

function line_style = get_line_style(noise_level)
    % Assign line styles based on noise level
    if noise_level <= 0.001
        line_style = '-';
    elseif noise_level <= 0.01
        line_style = '--';
    elseif noise_level <= 0.1
        line_style = ':';
    else
        line_style = '-.';
    end
end

function sensitivity = assess_noise_sensitivity(layer_data)
    % Assess overall noise sensitivity for a layer
    global_min_rate = sum(layer_data.ReachedGlobalMin) / height(layer_data) * 100;
    
    if global_min_rate >= 80
        sensitivity = 'Low (Robust)';
    elseif global_min_rate >= 50
        sensitivity = 'Medium';
    else
        sensitivity = 'High (Sensitive)';
    end
end

function analyze_weight_space_distances(original_params, final_params_all, results)
    % Analyze distances in weight space
    if isempty(final_params_all)
        text(0.5, 0.5, 'No final parameters available', 'HorizontalAlignment', 'center');
        title('Weight Space Analysis - No Data');
        return;
    end
    
    original_weights = extract_weights_vector(original_params);
    distances = [];
    noise_levels = [];
    
    for i = 1:length(final_params_all)
        if i <= height(results) && results.NoiseLevel(i) > 0
            final_weights = extract_weights_vector(final_params_all{i});
            distance = norm(final_weights - original_weights) / norm(original_weights);
            distances = [distances; distance];
            noise_levels = [noise_levels; results.NoiseLevel(i)];
        end
    end
    
    if ~isempty(distances)
        scatter(noise_levels, distances, 100, 'filled');
        xlabel('Noise Level');
        ylabel('Relative Weight Distance');
        title('Weight Space Distance from Original');
        grid on;
        set(gca, 'XScale', 'log');
    else
        text(0.5, 0.5, 'No valid weight data', 'HorizontalAlignment', 'center');
        title('Weight Space Analysis - No Valid Data');
    end
end

function weights_vector = extract_weights_vector(params)
    % Flatten all parameters into a single vector
    if isstruct(params)
        weights_vector = [params.We1(:); params.be1(:); params.We_latent(:); params.be_latent(:); ...
                         params.Wd1(:); params.bd1(:); params.Wd_output(:); params.bd_output(:)];
    else
        weights_vector = [];
    end
end

function loss = evaluate_loss(params, X_train, relu_func)
    % Quick loss evaluation
    total_loss = 0;
    num_samples = min(50, size(X_train, 1));
    
    for i = 1:num_samples
        X_sample = X_train(i, :);
        
        % Forward pass
        H1_enc = X_sample * params.We1 + params.be1;
        H1_enc_act = relu_func(H1_enc);
        Z = H1_enc_act * params.We_latent + params.be_latent;
        Z_act = relu_func(Z);
        H1_dec = Z_act * params.Wd1 + params.bd1;
        H1_dec_act = relu_func(H1_dec);
        X_recon = H1_dec_act * params.Wd_output + params.bd_output;
        
        error = X_recon - X_sample;
        sample_loss = 0.5 * sum(error.^2);
        total_loss = total_loss + sample_loss;
    end
    
    loss = total_loss / num_samples;
end