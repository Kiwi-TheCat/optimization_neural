function mse = compute_reconstruction_mse(params, X, relu)
    % normalized signal fed through the autoencoder
    mse = 0;
    N = size(X,1);
    for i=1:N
        x = X(i,:);
        h1 = relu(x * params.We1 + params.be1);
        z = relu(h1 * params.We_latent + params.be_latent);
        h2 = relu(z * params.Wd1 + params.bd1);
        x_hat = h2 * params.Wd_output + params.bd_output;
        % mse calculation
        mse = mse + mean((x - x_hat).^2);
    end
    mse = mse/N;
end