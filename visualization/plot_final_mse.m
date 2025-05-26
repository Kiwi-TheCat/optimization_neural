function plot_final_mse(mse_all, optimizers)
%PLOT_FINAL_MSE Displays bar chart of final reconstruction MSE per optimizer

figure('Name', 'Final Reconstruction MSE');
bar(mse_all);
set(gca, 'XTickLabel', upper(optimizers));
xlabel('Optimizer');
ylabel('Reconstruction MSE');
title('Optimizer Comparison - Final MSE');
grid on;
end
