function [loss, grads, x_hat] = forward_backward_pass(x, params, relu, relu_deriv)
    % === Forward Pass ===
    h1 = relu(x * params.We1 + params.be1);
    z = relu(h1 * params.We_latent + params.be_latent);
    h2 = relu(z * params.Wd1 + params.bd1);
    x_hat = h2 * params.Wd_output + params.bd_output;

    % === Loss ===
    loss = 0.5 * sum((x_hat - x).^2);

    % === Forward-only mode ===
    if isempty(relu_deriv)
        grads = struct();  % No gradients needed
        return;
    end

    % === Backward Pass ===
    dL = x_hat - x;
    
    grads.Wd_output = h2' * dL;
    grads.bd_output = dL;

    d_h2 = (dL * params.Wd_output') .* relu_deriv(h2);
    grads.Wd1 = z' * d_h2;
    grads.bd1 = d_h2;

    d_z = (d_h2 * params.Wd1') .* relu_deriv(z);
    grads.We_latent = h1' * d_z;
    grads.be_latent = d_z;

    d_h1 = (d_z * params.We_latent') .* relu_deriv(h1);
    grads.We1 = x' * d_h1;
    grads.be1 = d_h1;
end
