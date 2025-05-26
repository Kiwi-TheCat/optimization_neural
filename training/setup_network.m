%% --- Modularized Setup and Training Logic ---
function [params, optim, relu, relu_deriv] = setup_network(input_size, hidden_size, latent_size)
    params = initialize_parameters(input_size, hidden_size, latent_size);
    optim = initialize_optimizer_state(params);
    relu = @(x) max(0, x);
    relu_deriv = @(a) double(a > 0);
end
