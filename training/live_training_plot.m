function [x_test_sample,x_hat]  = live_training_plot(X_train, params, epoch, relu)
    figure(100); clf;
     
    % --- Test one encoding ---
    x_test_sample = X_train(1,:);
    h1 = relu(x_test_sample * params.We1 + params.be1);
    z = relu(h1 * params.We_latent + params.be_latent);

    % --- Final decoding for visualization ---
    h2 = relu(z * params.Wd1 + params.bd1);
    x_hat = h2 * params.Wd_output + params.bd_output;
    
    h1 = plot(x_test_sample, 'b'); hold on;
    h2 = plot(x_hat, 'r'); hold off;

    legend([h1, h2], {'Original', 'Reconstructed'});
    xlabel('Feature'); ylabel('Voltage');
    title(sprintf('Epoch %d: Original vs. Reconstructed', epoch));
    drawnow;
end