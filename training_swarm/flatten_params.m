function [vec, unpack] = flatten_params(params)
    fields = fieldnames(params);
    all_vals = [];
    shapes = struct();
    for i = 1:numel(fields)
        val = params.(fields{i});
        shapes.(fields{i}) = size(val);
        all_vals = [all_vals; val(:)];
    end
    vec = all_vals;

    % Unpacker as a nested function
    function out_params = unpack_fn(vector)
        idx = 1;
        out_params = struct();
        for i = 1:numel(fields)
            shape = shapes.(fields{i});
            len = prod(shape);
            out_params.(fields{i}) = reshape(vector(idx:idx+len-1), shape);
            idx = idx + len;
        end
    end
    unpack = @unpack_fn;
end
