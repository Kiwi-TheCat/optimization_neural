function mse = compute_reconstruction_mse(params, X_original, mean_X, std_X, relu)
    x = X_original(1, :);
    x_norm = (x - mean_X) ./ std_X;
    h1 = relu(x_norm * params.We1 + params.be1);
    z = relu(h1 * params.We_latent + params.be_latent);
    h2 = relu(z * params.Wd1 + params.bd1);
    x_hat_norm = h2 * params.Wd_output + params.bd_output;
    x_hat = x_hat_norm .* std_X + mean_X;
    mse = mean((x - x_hat).^2);
end