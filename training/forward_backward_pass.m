function [loss, grads, x_hat] = forward_backward_pass(x, params, relu, relu_deriv)
%FORWARD_BACKWARD_PASS Perform forward and backward pass through a 3-layer autoencoder.
%   Now supports batch input (N x D).
%
% Inputs:
%   x           - (N x D) input batch (N = batch size)
%   params      - struct with fields:
%                   We1        (D x H1)
%                   be1        (1 x H1)
%                   We_latent  (H1 x Z)
%                   be_latent  (1 x Z)
%                   Wd1        (Z x H2)
%                   bd1        (1 x H2)
%                   Wd_output  (H2 x D)
%                   bd_output  (1 x D)
%   relu        - activation function handle
%   relu_deriv  - derivative function handle (or [] for forward-only mode)
%
% Outputs:
%   loss        - scalar mean squared error over batch
%   grads       - struct of parameter gradients
%   x_hat       - (N x D) reconstructed outputs

    N = size(x, 1);  % Batch size

    % === Forward Pass ===
    h1 = relu(x * params.We1 + params.be1);                      % (N x H1)
    z  = relu(h1 * params.We_latent + params.be_latent);         % (N x Z)
    h2 = relu(z * params.Wd1 + params.bd1);                      % (N x H2)
    x_hat = h2 * params.Wd_output + params.bd_output;            % (N x D)

    % === Loss: Mean squared error averaged over batch ===
    reconstruction_error = x_hat - x;                            % (N x D)
    loss = 0.5 * mean(sum(reconstruction_error.^2, 2));          % scalar

    % === Forward-only mode ===
    if isempty(relu_deriv)
        grads = struct();  % No gradients needed
        return;
    end

    % === Backward Pass ===
    dL = reconstruction_error;                                   % (N x D)

    % Output layer
    grads.Wd_output = h2' * dL / N;                              % (H2 x D)
    grads.bd_output = sum(dL, 1) / N;                            % (1 x D)

    % Decoder hidden layer
    d_h2 = (dL * params.Wd_output') .* relu_deriv(h2);           % (N x H2)
    grads.Wd1 = z' * d_h2 / N;                                   % (Z x H2)
    grads.bd1 = sum(d_h2, 1) / N;                                % (1 x H2)

    % Latent layer
    d_z = (d_h2 * params.Wd1') .* relu_deriv(z);                 % (N x Z)
    grads.We_latent = h1' * d_z / N;                             % (H1 x Z)
    grads.be_latent = sum(d_z, 1) / N;                           % (1 x Z)

    % Encoder hidden layer
    d_h1 = (d_z * params.We_latent') .* relu_deriv(h1);          % (N x H1)
    grads.We1 = x' * d_h1 / N;                                   % (D x H1)
    grads.be1 = sum(d_h1, 1) / N;                                % (1 x H1)
end
