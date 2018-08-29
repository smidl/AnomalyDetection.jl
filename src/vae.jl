###########################
### vae NN construction ###
###########################

"""
	VAE{encoder, sampler, decoder, variant}

Flux-like structure for the variational autoencoder.
"""
struct VAE{E, S, D, V<:Val}
	encoder::E
	sampler::S
	decoder::D
	variant::V
end

VAE(E,S,D,V::Symbol = :unit) = VAE(E,S,D,Val(V))

# make the struct callable
(vae::VAE)(X) = vae.decoder(vae.sampler(vae.encoder(X)))

# and make it trainable
Flux.treelike(VAE)

"""
	mu(X)

Extract mean as the first horizontal half of X.
"""
mu(X) = X[1:Int(size(X,1)/2),:]

"""
	sigma2(X)

Extract sigma^2 as the second horizontal half of X. 
"""
sigma2(X) = softplus(X[Int(size(X,1)/2+1):end,:]) + Float(1e-6)

"""
	logps(x)

Is the logarithm of the standard pdf of x.
"""
logps(x) = abs.(-1/2*x.^2 - 1/2*log(2*pi))

"""
	samplenormal(X)

Sample normal distribution with mean and sigma2 extracted from X.
"""
function samplenormal(X)
	μ, σ2 = mu(X), sigma2(X)
	ϵ = Float.(randn(size(μ)))
	return μ .+  ϵ .* sqrt.(σ2)
end

"""
	VAE(esize, dsize; [activation, layer])

Initialize a variational autoencoder with given encoder size and decoder size.

esize - vector of ints specifying the width anf number of layers of the encoder
\ndsize - size of decoder
\nactivation [Flux.relu] - arbitrary activation function
\nlayer [Flux.Dense] - type of layer
\nvariant [:unit] - :unit - output has unit variance
\n 		          - :sigma - the variance of the output is estimated
"""
function VAE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu,
		layer = Flux.Dense, variant = :unit)
	@assert variant in [:unit, :sigma]
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == 2*dsize[1]
	(variant==:unit)? (@assert esize[1] == dsize[end]) :
		(@assert esize[1]*2 == dsize[end])

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	vae = VAE(encoder, samplenormal, decoder, variant)

	return vae
end

################
### training ###
################

"""
	KL(μ, σ2)

KL divergence between a normal distribution and unit gaussian.
"""
KL(μ, σ2) = Float(1/2)*mean(sum(σ2 + μ.^2 - log.(σ2) - 1, 1))

"""
	KL(vae, X)

KL divergence between the encoder output and unit gaussian.
"""
function KL(vae::VAE, X) 
	ex = vae.encoder(X)
	KL(mu(ex), sigma2(ex))
end

"""
	likelihood(X, μ, [σ2])

Likelihood of a sample X given mean and variance.
"""
likelihood(X, μ) = - mean(sum((μ - X).^2,1))/2
likelihood(X, μ, σ2) = - mean(sum((μ - X).^2./σ2 + log.(σ2),1))/2

"""
	likelihood(vae, X)

Likelihood of an autoencoded sample X.
"""
function likelihood(vae::VAE{E,S,D,V}, X) where {E,S,D,V<:Val{:sigma}}
	vx = vae(X)
	μ, σ2 = mu(vx), sigma2(vx)
	return likelihood(X,μ, σ2)
end
function likelihood(vae::VAE{E,S,D,V}, X) where {E,S,D,V<:Val{:unit}}
	μ = vae(X)
	return likelihood(X,μ)
end
"""
	likelihood(vae, X, M)

Likelihood of an autoencoded sample X sampled M times.
"""
likelihood(vae::VAE, X, M) = mean([likelihood(vae, X) for m in 1:M])

"""
	loss(vae, X, M, lambda)

Loss function of the variational autoencoder. Lambda is scaling parameter of
the KLD, 1 = full KL, 0 = no KL.
"""
loss(vae::VAE, X, M, lambda) = -likelihood(vae,X,M) + Float(lambda)*KL(vae, X)

"""
	evalloss(vae, X, M, lambda)

Print vae loss function values.
"""
function evalloss(vae::VAE, X, M, lambda) 
	l, lk, kl = getlosses(vae, X, M, lambda)
	print("loss: ", l,
	"\nlikelihood: ", lk,
	"\nKL: ", kl, "\n\n")
end

"""
	getlosses(vae, X, M, lambda)

Return the numeric values of current losses.
"""
getlosses(vae::VAE, X, M, lambda) = (
	Flux.Tracker.data(loss(vae, X, M, lambda)),
	Flux.Tracker.data(-likelihood(vae,X,M)),
	Flux.Tracker.data(KL(vae, X))
	)

"""
	fit!(vae, X, batchsize, [M, iterations, cbit, nepochs, 
	verb, lambda, rdelta, history, eta])

Trains the VAE neural net.

vae - a VAE object
\nX - data array with instances as columns
\nbatchsize - batchsize
\nM [1] - number of samples for likelihood
\niterations [1000] - number of iterations
\ncbit [200] - after this # of iterations, output is printed
\nnepochs [nothing] - if this is supplied, epoch training will be used instead of fixed iterations
\nverb [true] - if output should be produced
\nlambda [1] - scaling for the KLD loss
\nrdelta [Inf] - stopping condition for likelihood
\nhistory [nothing] - a dictionary for training progress control
\neta [eta] - learning rate
"""
function fit!(vae::VAE, X, batchsize; M=1, iterations=1000, cbit = 200, 
	nepochs = nothing, verb::Bool = true, lambda = 1,
	rdelta = Inf, history = nothing, eta = 0.001)
	# settings
	opt = ADAM(params(vae), eta)

	# sampler
	# sampler
	if nepochs == nothing
		sampler = UniformSampler(X,iterations,batchsize)
	else
		sampler = EpochSampler(X,nepochs,batchsize)
		cbit = sampler.epochsize
		iterations = nepochs*cbit
	end
	# it might be smaller than the original one if there is not enough data
	batchsize = sampler.batchsize 

	# using ProgressMeter 
	if verb
		p = Progress(iterations, 0.3)
		x = next!(sampler)
		reset!(sampler)
		_l, _lk, _kl = getlosses(vae, x, M, lambda)
	end

	# train
	for (i,x) in enumerate(sampler)
		# gradient computation and update
		l = loss(vae, x, M, lambda)
		Flux.Tracker.back!(l)
		opt()

		# progress
		if verb 
			if (i%cbit == 0 || i == 1)
				_l, _lk, _kl = getlosses(vae, x, M, lambda)
			end
			ProgressMeter.next!(p; showvalues = [(:loss,_l),(:likelihood, _lk),(:KL, _kl)])
		end

		# save actual iteration data
		if history != nothing
			track!(vae, history, x, M, lambda)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = Flux.Tracker.data(-likelihood(vae, x))[1]
			if re < rdelta
				println("Training ended prematurely after $i iterations,\n",
					"likelihood $re < $rdelta")
				break
			end
		end
	end
end

"""
	track!(vae, history, X, M, lambda)

Save current progress.
"""
function track!(vae::VAE, history::MVHistory, X, M, lambda)
	l, lk, kl = getlosses(vae, X, M, lambda)
	push!(history, :loss, l)
	push!(history, :KLD, lk)
	push!(history, :likelihood, kl)
end


##################
### vae output ###
##################

"""
	getcode(vae, X)

Produces code z for given X.
"""
getcode(vae::VAE, X) = vae.sampler(vae.encoder(X))

"""
	generate(vae, [n])

Generate a sample from the posterior.
"""
generate(vae::VAE{E,S,D,V}, n::Int = 1) where {E,S,D,V<:Val{:unit}} = vae.decoder(Float.(randn(Int(size(vae.encoder.layers[end].W,1)/2),n))).data
generate(vae::VAE{E,S,D,V}, n::Int = 1) where {E,S,D,V<:Val{:sigma}} =
	 samplenormal(vae.decoder(Float.(randn(Int(size(vae.encoder.layers[end].W,1)/2),n))).data)


######################
### classification ###
######################

"""
	anomalyscore(vae, X, M, [t])

Compute anomaly score for X.

t = type, default "likelihood", otherwise "logpn".
"""
anomalyscore(vae::VAE, X::Array{Float, 1}, M, t = "likelihood") =
	(t=="likelihood")? Flux.Tracker.data(-mean([likelihood(vae, X) for i in 1:M])) : mean(logps(Flux.Tracker.data(getcode(vae,X))))
anomalyscore(vae::VAE, X::Array{Float, 2}, M, t = "likelihood") =
	reshape(mapslices(y -> anomalyscore(vae, y, M, t), X, 1), size(X,2))

"""
	classify(vae, x, threshold, M)

Classify an instance x using likelihood and threshold.
"""
classify(vae::VAE, X, threshold, M) = Int.(anomalyscore(vae, X, M) .> threshold)


##############################################################################
### A SK-learn like model based on VAE with the same methods and some new. ###
##############################################################################

"""
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct VAEmodel <: genmodel
	vae::VAE
	lambda::Real
	threshold::Real
	contamination::Real
	iterations::Int
	cbit::Real
	nepochs
	verbfit::Bool
	batchsize::Int # batchsize
	M::Int # sampling rate for likelihood
	rdelta::Float
	Beta::Float
	history
	astype
	eta::Real
end

"""
	VAEmodel(esize, dsize, [lambda, threshold, contamination, iterations,
	batchsize,  verbfit, cbit, nepochs,
	M, activation, layer, rdelta, Beta, tracked, astype, eta])

Initialize a variational autoencoder model with given parameters.

esize - encoder architecture, e.g. [input_dim, 10, z_dim*2]
\ndsize - decoder architecture, e.g. [z_dim, 10, input_dim]
\nlambda [1e-4] - weight of the KL divergence in the total loss
\nthreshold [0.0] - anomaly score threshold for classification, is set automatically using contamination during fit
\ncontamination [0.0] - percentage of anomalous samples in all data for automatic threshold computation
\niterations [10000] - number of training iterations
\ncbit [1000] - current training progress is printed every cbit iterations
\nnepochs [nothing] - if this is supplied, epoch training will be used instead of fixed iterations
\nverbfit [true] - is progress printed?
\nbatchsize [256] - batchsize
\nM [1] - number of samples taken during computation of likelihood, higher may produce more stable classification results
\nactivation [Flux.relu] - activation function
\nlayer [Flux.dense] - layer type
\nrdelta [Inf] - training stops if likelihood is smaller than rdelta
\nBeta [1.0] - how tight around normal data is the automatically computed threshold
\ntracked [false] - is training progress (losses) stored?
\nastype ["likelihood"] - type of anomaly score function
\nvariant [:unit] - :unit - output has unit variance
\n		          - :sigma - the variance of the output is estimated
\neta [0.001] - learning rate of the optimizer
"""
function VAEmodel(esize::Array{Int64,1}, dsize::Array{Int64,1};
	contamination::Real = 0.0, iterations::Int = 10000,
	cbit::Int=1000, nepochs = nothing,
	verbfit::Bool=true, batchsize::Int=256, 
	lambda::Real = 1e-4, threshold::Real = 0.0, 
	M::Int =1, activation = Flux.relu,
	layer = Flux.Dense, rdelta = Inf, Beta = 1.0, tracked = false,
	astype = "likelihood", variant = :unit, eta = 0.001)
	# construct the AE object
	vae = VAE(esize, dsize, activation = activation, layer = layer, variant = variant)
	(tracked)? history = MVHistory() : history = nothing
	model = VAEmodel(vae, lambda, threshold, contamination, iterations, cbit, 
		nepochs, verbfit,
		batchsize, M, rdelta, Beta, history, astype, eta)
	return model
end

# reimplement some methods of VAE
(model::VAEmodel)(x) = model.vae(x)
muz(model::VAEmodel, X) = mu(model.vae.encoder(X))
sigma2z(model::VAEmodel, X) = sigma2(model.vae.encoder(X))
mux(model::VAEmodel, X) = (model.vae.variant == Val{:unit}())? model(X) : mu(model(X))
sigma2x(model::VAEmodel, X) = (model.vae.variant == Val{:unit}())? nothing : sigma2(model(X))
sample_z(model::VAEmodel, X) = samplenormal(model.vae.encoder(X))
getcode(model::VAEmodel, X) = getcode(model.vae, X)
KL(model::VAEmodel, X) = KL(model.vae, X)
likelihood(model::VAEmodel, X) = likelihood(model.vae, X)
loss(model::VAEmodel, X) = loss(model.vae, X, model.M, model.lambda)
evalloss(model::VAEmodel, X) = evalloss(model.vae, X, model.M, model.lambda)
getlosses(model::VAEmodel, X) = getlosses(model.vae, X, model.M, model.lambda)
generate(model::VAEmodel) = generate(model.vae)
generate(model::VAEmodel, n::Int) = generate(model.vae, n)
classify(model::VAEmodel, x) = classify(model.vae, x, model.threshold, model.M)
getthreshold(model::VAEmodel, x) = getthreshold(model.vae, x, model.contamination, model.M; Beta = model.Beta)
anomalyscore(model::VAEmodel, X) = anomalyscore(model.vae, X, model.M, model.astype)
params(model::VAEmodel) = Flux.params(model.vae)

"""
	setthreshold!(model::VAEmodel, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::VAEmodel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model, X, Y)

Fit the VAE model, instances are columns of X.
"""
function fit!(model::VAEmodel, X)
	# fit the VAE NN
	fit!(model.vae, X, model.batchsize, M = model.M, iterations = model.iterations,
	cbit = model.cbit, nepochs = model.nepochs,
	verb = model.verbfit, lambda = model.lambda,
	rdelta = model.rdelta, history = model.history, eta = model.eta)
end

"""
	predict(model, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::VAEmodel, X)
	return classify(model, X)
end
