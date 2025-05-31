function [params, optim, act, act_deriv] = setup_network(input_size, hidden_size, latent_size, act_type)

    if nargin < 4, act_type = 'relu'; end
    
    params = initialize_parameters(input_size, hidden_size, latent_size);
    optim = initialize_optimizer_state(params);

    switch lower(act_type)
        case 'relu'
            act = @(x) max(0, x);
            act_deriv = @(a) double(a > 0);
        case 'sigmoid'
            act = @(x) 1 ./ (1 + exp(-x));
            act_deriv = @(a) a .* (1 - a);  % assumes forward already passed through sigmoid
        case 'tanh'
            act = @(x) tanh(x);
            act_deriv = @(a) 1 - a.^2;
        otherwise
            error('Unsupported activation type');
    end
end
