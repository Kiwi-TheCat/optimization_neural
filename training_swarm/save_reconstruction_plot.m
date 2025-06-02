function save_reconstruction_plot(x_original, x_reconstructed, baseName)
% SAVE_RECONSTRUCTION_PLOT Plots and saves the reconstruction vs original signal.
%
% Usage:
%   save_reconstruction_plot(x_original, x_reconstructed, 'reconstruction_plot')
%
% Inputs:
%   x_original      - Original input signal (1D vector)
%   x_reconstructed - Reconstructed signal (1D vector)
%   baseName        - Base filename (without extension)

    if nargin < 3
        baseName = 'reconstruction_plot';
    end

    % Plot signals
    figure('Name', 'Reconstruction Comparison', 'NumberTitle', 'off');
    h1 = plot(x_original, 'b', 'LineWidth', 1.5); hold on;
    h2 = plot(x_reconstructed, 'r--', 'LineWidth', 1.5);
    
    % Add legend safely
    if isgraphics(h1) && isgraphics(h2(1))
        legend([h1, h2], {'Original', 'Reconstructed'}, 'Location', 'best');
    end
    
    xlabel('Index');
    ylabel('Value');
    title('Original vs Reconstructed Signal');
    grid on;

    % === Generate Unique Filename ===
    ext = '.png';
    filename = [baseName, ext];
    counter = 1;

    while exist(filename, 'file')
        filename = sprintf('%s_%d%s', baseName, counter, ext);
        counter = counter + 1;
    end

    saveas(gcf, filename);
    fprintf('Saved plot as: %s\n', filename);

    close(gcf);  % Optional: close figure to save resources
end
