function params = initialize_parameters(input_size, hidden_size, latent_size)

    rng(0);  % <- Set random seed for reproducibility
    
    params.We1 = randn(input_size, hidden_size) * sqrt(2 / input_size); % this creates a 
    params.be1 = zeros(1, hidden_size);

    params.We_latent = randn(hidden_size, latent_size) * sqrt(2 / hidden_size);
    params.be_latent = zeros(1, latent_size);

    params.Wd1 = randn(latent_size, hidden_size) * sqrt(2 / latent_size);
    params.bd1 = zeros(1, hidden_size);

    params.Wd_output = randn(hidden_size, input_size) * sqrt(2 / hidden_size);
    params.bd_output = zeros(1, input_size);

end
