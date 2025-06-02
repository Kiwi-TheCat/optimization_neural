function [loss, grads, x_hat] = forward_backward_pass(x, params, relu, relu_deriv)
%FORWARD_BACKWARD_PASS Perform a forward and backward pass through a 3-layer autoencoder.
%
%   Inputs:
%       x           - (1 x D) input sample vector
%       params      - struct containing weight and bias matrices:
%                       .We1        (D x H1)    encoder layer 1 weights
%                       .be1        (1 x H1)    encoder layer 1 biases
%                       .We_latent  (H1 x Z)    latent encoder weights
%                       .be_latent  (1 x Z)     latent encoder biases
%                       .Wd1        (Z x H2)    decoder layer weights
%                       .bd1        (1 x H2)    decoder layer biases
%                       .Wd_output  (H2 x D)    output weights
%                       .bd_output  (1 x D)     output biases
%       relu        - handle to activation function (e.g., ReLU or leaky ReLU)
%       relu_deriv  - handle to derivative of activation function
%
%   Outputs:
%       loss        - scalar reconstruction loss (Mean Squared Error)
%       grads       - struct containing gradients for all weights and biases
%       x_hat       - (1 x D) reconstructed output

    % === Forward Pass ===
    h1 = relu(x * params.We1 + params.be1);                      % Encoder hidden layer
    z = relu(h1 * params.We_latent + params.be_latent);          % Latent layer (bottleneck)
    h2 = relu(z * params.Wd1 + params.bd1);                      % Decoder hidden layer
    x_hat = h2 * params.Wd_output + params.bd_output;            % Output layer (reconstruction)

    % === Loss (Mean Squared Error) ===
    % The factor 0.5 is used for mathematical convenience:
    % When deriving the gradient of 0.5 * (x_hat - x)^2, the 2 cancels out:
    %   d/dx [0.5 * (x - y)^2] = x - y
    loss = 0.5 * sum((x_hat - x).^2);  % scalar loss
    
    % === Forward-only mode ===
    if isempty(relu_deriv)
        grads = struct();  % No gradients needed
        return;
    end
    % === Backward Pass ===
    % Gradient of the loss with respect to output
    dL = x_hat - x;

    % Output layer gradients
    grads.Wd_output = h2' * dL;        % (H2 x D)
    grads.bd_output = dL;              % (1 x D)

    % Decoder hidden layer gradients
    d_h2 = (dL * params.Wd_output') .* relu_deriv(h2);  % (1 x H2)
    grads.Wd1 = z' * d_h2;               % (Z x H2)
    grads.bd1 = d_h2;                    % (1 x H2)

    % Latent layer gradients
    d_z = (d_h2 * params.Wd1') .* relu_deriv(z);       % (1 x Z)
    grads.We_latent = h1' * d_z;         % (H1 x Z)
    grads.be_latent = d_z;               % (1 x Z)

    % Encoder gradients
    d_h1 = (d_z * params.We_latent') .* relu_deriv(h1);   % (1 x H1)
    grads.We1 = x' * d_h1;             % (D x H1)
    grads.be1 = d_h1;                  % (1 x H1)
end
