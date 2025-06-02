function plot_final_mse(mse_all, optimizers)
    % PLOT_FINAL_MSE - Displays a bar chart of final reconstruction MSE per optimizer
    % INPUTS:
    %   mse_all    - vector of final MSE values
    %   optimizers - cell array of optimizer names
    
    figure('Name', 'Final Reconstruction MSE', 'Color', 'w');
    
    % Plot the bar chart with value annotations
    b = bar(mse_all, 'FaceColor', [0.2, 0.6, 0.8]);
    hold on;

    % Annotate bars with exact MSE values
    xtips = b.XEndPoints;
    ytips = b.YEndPoints;
    labels = string(round(b.YData, 4));
    text(xtips, ytips + max(mse_all) * 0.01, labels, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 9);
    
    % X-axis labels
    set(gca, 'XTick', 1:numel(optimizers), 'XTickLabel', upper(optimizers));
    xtickangle(30);

    % Labels and formatting
    xlabel('Optimizer', 'FontWeight', 'bold');
    ylabel('Final Reconstruction MSE', 'FontWeight', 'bold');
    title('Comparison of Optimizers by Final MSE', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    box off;
end
