function snapshot = save_epoch_weights(params)
    snapshot.We1        = params.We1;
    snapshot.be1        = params.be1;
    
    snapshot.We_latent  = params.We_latent;
    snapshot.be_latent  = params.be_latent;
    
    snapshot.Wd1        = params.Wd1;
    snapshot.bd1        = params.bd1;
    
    snapshot.Wd_output  = params.Wd_output;
    snapshot.bd_output  = params.bd_output;
end
