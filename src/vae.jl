using Flux


"""
	VAE{encoder, sampler, decoder}

Flux-like structure for the variational autoencoder.
"""
struct VAE{E, S, D}
	encoder::E
	sampler::S
	decoder::D
end

# make the struct callable
(model::VAE)(x) = model.decoder(model.sampler(model, model.encoder(x)))

# and make it trainable
Flux.treelike(VAE)

""" 
	softplus(x)

softplus(x) = log(exp(x) + 1)	
"""
softplus(x) = log.(exp.(x)+1)

"""
	mu(model::VAE, y)

Extract means from the last encoder layer.
"""
mu(model::VAE, y) = y[1:Int(size(model.encoder.layers[end].W,1)/2),:]

"""
	sigma(model::VAE, y)

Extract sigmas from the last encoder layer.
"""
sigma(model::VAE, y) = softplus(y[Int(size(model.encoder.layers[end].W,1)/2+1):end,:]) + 1e-6

"""
	sample_z(model::VAE, y)

Sample from the last encoder layer.
"""
sample_z(model::VAE, y) = randn(size(mu(model, y))) .* sigma(model,y) + mu(model,y)

"""
	VAE(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int)


"""
function VAE(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int)
	# construct the encoder
	encoder = Dense(indim,hiddendim,Flux.relu)
	for i in 2:nlayers
	    encoder = Chain(encoder, Dense(hiddendim,hiddendim,Flux.relu))
	end
	encoder = Chain(encoder, Dense(hiddendim, 2*latentdim))
	    
	# construct the decoder
	decoder = Dense(latentdim, hiddendim, Flux.relu)
	for i in 2:nlayers
	    decoder = Chain(decoder, Dense(hiddendim, hiddendim, Flux.relu))
	end
	decoder = Chain(decoder, Dense(hiddendim, indim))    

	# finally construct the model struct
	model = VAE(encoder, sample_z, decoder)

	return model
end


# loss functions 
"""
	KL(model::VAE, x)

KL divergence between the encoder parameters and unit gaussian.
"""
KL(model::VAE, x) = 1/2*sum(sigma(model, model.encoder(x)).^2 + mu(model, model.encoder(x)).^2 - log.(sigma(model, model.encoder(x)).^2) - 1)

"""
	rerr(model::VAE, x, y)

Reconstruction error.
"""
rerr(model::VAE, x, y) = Flux.mse(model(x), y) # reconstruction error, not for training

"""
	loss(model::VAE, x,y)

Loss function of the variational autoencoder.
"""
loss(model::VAE, x,y) = rerr(model, x, y) + KL(model, x)

"""
	fit!(model::VAE, x)

Trains the VAE model.
"""
function fit!(model::VAE, x; iterations=1000, throttle = 5)
	# settings
	opt = ADAM(params(model))
	dataset = repeated((model, x, x), iterations) # y=x
	
	# callback
	evalcb = () -> print("loss: ", loss(model, x, x), "\nreconstruction error: ", rerr(model, x, x), "\nKL: ", KL(model, x), "\n\n")
	cb = Flux.throttle(evalcb, throttle)

	# train
	Flux.train!(loss, dataset, opt, cb = cb)
end

"""
	generate_sample(model::VAE)

Generate a sample from the posterior.
"""
function generate_sample(model::VAE)
	return model.decoder(randn(1))
end

"""
	generate_sample(model::VAE, n::Int)

Generate n samples.
"""
function generate_sample(model::VAE, n::Int)
	return model.decoder(randn(1,n))
end