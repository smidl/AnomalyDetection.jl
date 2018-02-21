using Flux
import Base.Iterators.repeated

### vae NN construction ###

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
(vae::VAE)(X) = vae.decoder(vae.sampler(vae, vae.encoder(X)))

# and make it trainable
Flux.treelike(VAE)

""" 
	softplus(X)

softplus(X) = log(exp(X) + 1)	
"""
softplus(X) = log.(exp.(X)+1)

"""
	mu(vae::VAE, X)

Extract means from the last encoder layer.
"""
mu(vae::VAE, X) = X[1:Int(size(vae.encoder.layers[end].W,1)/2),:]

"""
	sigma(vae::VAE, X)

Extract sigmas from the last encoder layer.
"""
sigma(vae::VAE, X) = softplus(X[Int(size(vae.encoder.layers[end].W,1)/2+1):end,:]) + 1e-6

"""
	sample_z(vae::VAE, X)

Sample from the last encoder layer.
"""
#sample_z(vae::VAE, X) = mu(vae,X)
sample_z(vae::VAE, X) = randn(size(mu(vae, X))) .* sigma(vae,X) + mu(vae,X)

"""
	VAE(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int)

Initialize a variational autoencoder with given parameters.
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

	# finally construct the vae struct
	vae = VAE(encoder, sample_z, decoder)

	return vae
end

### fitting ###

"""
	KL(vae::VAE, X)

KL divergence between the encoder parameters and unit gaussian.
"""
KL(vae::VAE, X) = 1/2*mean(sum(sigma(vae, vae.encoder(X)).^2 + mu(vae, vae.encoder(X)).^2 - log.(sigma(vae, vae.encoder(X)).^2) - 1, 1))

"""
	rerr(vae::VAE, X, Y)

Reconstruction error.
"""
rerr(vae::VAE, X) = Flux.mse(vae(X), X) # reconstruction error, not for training

"""
	loss(vae::VAE, X, lambda)

Loss function of the variational autoencoder. Lambda is scaling parameter of 
the reconstruction loss, lambda = 1/sigma^2 in p(x|z).
"""
loss(vae::VAE, X, lambda) = rerr(vae, X) + lambda*KL(vae, X)

"""
	evalcb(vae::VAE, X, lambda)

Print vae loss function values.	
"""
evalloss(vae::VAE, X, lambda) = print("loss: ", loss(vae, X, lambda).data[1], "\nreconstruction error: ", rerr(vae, X).data[1], "\nKL: ", KL(vae, X).data[1], "\n\n")

"""
	fit!(vae::VAE, X; iterations=1000, throttle = 5, verb = true, lambda = 1)

Trains the VAE neural net.
"""
function fit!(vae::VAE, X; iterations=1000, throttle = 5, verb = true, lambda = 1)
	# settings
	opt = ADAM(params(vae))
	if iterations != 0
		dataset = repeated((vae, X, lambda), iterations) # Y=x
		evalcb = () -> print("loss: ", loss(vae, X, lambda).data[1], "\nreconstruction error: ", rerr(vae, X).data[1], "\nKL: ", KL(vae, X).data[1], "\n\n")	
	else
		dataset = X # if x is already an iterable to be trained on
		evalcb = () -> print("loss: ", loss(vae, X[1][2], X[1][3]).data[1], "\nreconstruction error: ", rerr(vae, X[1][2]).data[1], "\nKL: ", KL(vae, X[1][2]).data[1], "\n\n")	
	end
	
	# callback
	cb = Flux.throttle(evalcb, throttle)

	# train
	if verb
		Flux.train!(loss, dataset, opt, cb = cb)
	else
		Flux.train!(loss, dataset, opt)
	end
end

### vae output ###

"""
	generate_sample(vae::VAE)

Generate a sample from the posterior.
"""
generate_sample(vae::VAE) = vae.decoder(randn(Int(size(vae.encoder.layers[end].W,1)/2))).data

"""
	generate_sample(vae::VAE, n::Int)

Generate n samples.
"""
generate_sample(vae::VAE, n::Int) = vae.decoder(randn(Int(size(vae.encoder.layers[end].W,1)/2),n)).data

"""
	classify(vae::VAE, x, threshold)

Classify an instance x using reconstruction error and threshold.
"""
classify(vae::VAE, x, threshold) = (rerr(vae, x) > threshold)? 1 : 0
classify(vae::VAE, x::Array{Float64,1}, threshold; L=1) = (rerr(vae, repmat(x,1,L)) > threshold)? 1 : 0
classify(vae::VAE, X::Array{Float64,2}, threshold; L=1) = reshape(mapslices(y -> classify(vae, y, threshold, L=L), X, 1), size(X,2))

"""
	get_threshold(vae::VAE, x, contamination)

Compute threshold for VAE classification based on known contamination level.
"""
function get_threshold(vae::VAE, x, contamination)
	N = size(x, 2)
	# get reconstruction errors
	xerr  = mapslices(y -> rerr(vae, y), x, 1)
	# create ordinary array from the tracked array
	xerr = reshape([e.data[1] for e in xerr], N)
	# sort it
	xerr = sort(xerr)
	aN = Int(floor(N*contamination)) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	return (xerr[end-aN]+xerr[end-aN+1])/2
end

### A SK-learn like model based on VAE with the same methods and some new. ###
""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct VAEmodel
	vae::VAE
	lambda::Real
	threshold::Real
	contamination::Real
	iterations::Int
	cbthrottle::Real
	verbfit::Bool
	L::Int # number of samples for classification
end

"""
	VAEmodel(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int,
	lambda::Real, threshold::Real, contamination::Real, iteration::Int, 
	cbthrottle::Real, verbfit::Bool)

Initialize a variational autoencoder model with given parameters.
"""
function VAEmodel(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int,
	lambda::Real, threshold::Real, contamination::Real, iterations::Int, 
	cbthrottle::Real, verbfit::Bool, L::Int)
	# construct the VAE object
	vae = VAE(indim, hiddendim, latentdim, nlayers)
	model = VAEmodel(vae, lambda, threshold, contamination, iterations, cbthrottle, verbfit, L)
	return model
end

# reimplement some methods of VAE
(model::VAEmodel)(x) = model.vae(x)   
mu(model::VAEmodel, X) = mu(model.vae, model.vae.encoder(X))
sigma(model::VAEmodel, X) = sigma(model.vae, model.vae.encoder(X))
sample_z(model::VAEmodel, X) = sample_z(model.vae, model.vae.encoder(X))
KL(model::VAEmodel, X) = KL(model.vae, X)
rerr(model::VAEmodel, X) = rerr(model.vae, X)
loss(model::VAEmodel, X) = loss(model.vae, X, model.lambda)
evalloss(model::VAEmodel, X) = evalloss(model.vae, X, model.lambda) 
generate_sample(model::VAEmodel) = generate_sample(model.vae)
generate_sample(model::VAEmodel, n::Int) = generate_sample(model.vae, n)
classify(model::VAEmodel, x) = classify(model.vae, x, model.threshold, L = model.L)
classify(model::VAEmodel, x, threshold) = classify(model.vae, x, threshold, L = model.L)
get_threshold(model::VAEmodel, x) = get_threshold(model.vae, x, model.contamination)
get_threshold(model::VAEmodel, x, contamination) = get_threshold(model.vae, x, contamination)

"""
	fit!(model::VAEmodel, X)

Fit the VAE model, instances are columns of X.	
"""
fit!(model::VAEmodel, X) = fit!(model.vae, X, iterations = model.iterations, 
	throttle = model.cbthrottle, verb = model.verbfit, lambda = model.lambda)

"""
	predict(model::VAEmodel, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::VAEmodel, X)
	model.threshold = get_threshold(model.vae, X, model.contamination)
	return classify(model.vae, X, model.threshold, L=model.L)
end