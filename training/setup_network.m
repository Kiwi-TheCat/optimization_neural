function [params, optim, relu, leaky_relu, relu_deriv, leaky_relu_deriv] = setup_network(input_size, hidden_size, latent_size)
    if nargin == 0
        % Return only activation functions
        params = [];
        optim = [];
        relu = @(x) max(0, x);
        relu_deriv = @(a) double(a > 0);
        alpha = 0.1;
        leaky_relu       = @(x, alpha) max(alpha .* x, x);
        leaky_relu_deriv = @(x, alpha) double(x >= 0) + alpha .* double(x < 0);
        return;
    end

    % Normal behavior with args
    params = initialize_parameters(input_size, hidden_size, latent_size);
    optim = initialize_optimizer_state(params);
    
    relu = @(x) max(0, x);
    relu_deriv = @(a) double(a > 0);

    alpha = 0.1;
    leaky_relu       = @(x, alpha) max(alpha .* x, x);
    leaky_relu_deriv = @(x, alpha) double(x >= 0) + alpha .* double(x < 0);
end