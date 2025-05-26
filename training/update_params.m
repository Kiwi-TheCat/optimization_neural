function [params, optim] = update_params(params, grads, optim, lr, method, t)
    beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
    fields = fieldnames(params);
    for i = 1:numel(fields)
        key = fields{i};
        g = grads.(key);
        switch method
            case 'sgd'
                params.(key) = params.(key) - lr * g;
            case 'adagrad'
                optim.cache.(key) = optim.cache.(key) + g.^2;
                params.(key) = params.(key) - lr * g ./ (sqrt(optim.cache.(key)) + eps);
            case 'adam'
                optim.m.(key) = beta1 * optim.m.(key) + (1 - beta1) * g;
                optim.v.(key) = beta2 * optim.v.(key) + (1 - beta2) * (g.^2);
                m_hat = optim.m.(key) / (1 - beta1^t);
                v_hat = optim.v.(key) / (1 - beta2^t);
                params.(key) = params.(key) - lr * m_hat ./ (sqrt(v_hat) + eps);
        end
    end
end