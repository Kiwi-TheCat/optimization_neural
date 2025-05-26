function plot_training_comparison(loss_all, optimizers, num_epochs)
    figure;
    plot(1:num_epochs, loss_all, 'LineWidth', 1.5);
    legend(upper(optimizers), 'Location', 'northeast');
    xlabel('Epoch');
    ylabel('Loss');
    title('Optimizer Comparison - Training Loss');
    grid on;
end