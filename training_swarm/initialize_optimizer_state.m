function optim = initialize_optimizer_state(params)
    fields = fieldnames(params);
    for i = 1:numel(fields)
        sz = size(params.(fields{i}));
        optim.m.(fields{i}) = zeros(sz);
        optim.v.(fields{i}) = zeros(sz);
        optim.cache.(fields{i}) = zeros(sz);
    end
end