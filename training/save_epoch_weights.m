function snapshot = save_epoch_weights(params,z)
    snapshot.We1 = params.We1;
    snapshot.We_latent = params.We_latent;
    snapshot.Wd1 = params.Wd1;
    snapshot.Wd_output = params.Wd_output;
    snapshot.z = z;
end