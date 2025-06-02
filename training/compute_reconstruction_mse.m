function mse = compute_reconstruction_mse(params, X, relu)
%COMPUTE_RECONSTRUCTION_MSE Computes mean squared error over a batch or single signal
%
%   Inputs:
%       params  - struct containing trained weights and biases
%       X       - (N x D) input matrix (N samples, D features)
%       relu    - activation function handle (e.g., @(x) max(0,x))
%
%   Output:
%       mse     - scalar average reconstruction loss over all samples

    % === Forward pass ===
    h1 = relu(X * params.We1 + params.be1);                       % (N x H1)
    z  = relu(h1 * params.We_latent + params.be_latent);          % (N x Z)
    h2 = relu(z * params.Wd1 + params.bd1);                       % (N x H2)
    x_hat = h2 * params.Wd_output + params.bd_output;             % (N x D)

    % === MSE over the entire dataset ===
    errors = x_hat - X;                                           % (N x D)
    mse = 0.5 * mean(sum(errors.^2, 2));                          % scalar
end
