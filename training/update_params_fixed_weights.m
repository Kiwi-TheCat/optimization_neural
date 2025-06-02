function [params, optim] = update_params_fixed_weights(params, grads, optim, lr, method, t, lambda)
    beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
    fields = fieldnames(params);
    % fields = {'We1'      }
    % {'be1'      }
    % {'We_latent'}
    % {'be_latent'}
    % {'Wd1'      }
    % {'bd1'      }
    % {'Wd_output'}
    % {'bd_output'}

    for i = 1:numel(fields)
        key = fields{i};
        g = grads.(key);
        
        rng(42);  % for reproducibility
        num_updates = 200; % used for masking
        total_elements = numel(g);  % Total number of scalar elements in gradients
        update_indices = randperm(total_elements, num_updates);  % 200 unique masking indices
        switch method
            case 'sgd'
                full_update = lr * g;
                full_update = full_update + lambda * params.(key);  % L2 penalty

            case 'adagrad'
                optim.cache.(key) = optim.cache.(key) + g.^2;
                % Full update for all entries
                full_update = lr * g ./ (sqrt(optim.cache.(key)) + eps);
                full_update = full_update + lambda * params.(key);  % L2 penalty

            case 'adam' % fixed weights only for the adam algorithm
                % m = first moment estimate (exponential moving average of the gradients)
                % v = second moment estimate (exponential moving average of the squared gradients)
                
                % === Adam update only at selected indices ===
                optim.m.(key) = beta1 * optim.m.(key) + (1 - beta1) * g;
                optim.v.(key) = beta2 * optim.v.(key) + (1 - beta2) * (g.^2);
                
                m_hat = optim.m.(key) / (1 - beta1^t);
                v_hat = optim.v.(key) / (1 - beta2^t);
                
                % Full update for all entries
                full_update = lr * m_hat ./ (sqrt(v_hat) + eps);
                full_update = full_update + lambda * params.(key);  % L2 penalty

        end
        % Masked update — only apply to selected indices
        masked_update = zeros(size(params.(key)));
        masked_update(update_indices) = full_update(update_indices);
                
        % Final parameter update
        params.(key) = params.(key) - masked_update;
    end
end