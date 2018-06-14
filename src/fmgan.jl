############################
### feature matching fmGAN ###
############################

"""
	fmGAN

Flux-like object representing a feature-matching generative adversarial network.
The architecture is the same as with ordinary fmGAN.
"""
struct fmGAN
	g # generator
	gg # non-trainable generator view
	d # discriminator
	dd # non-trainable discriminator view
	fmd # discriminator view for feature matching (first to last-but-one layers)
	pz # code distribution
end

# make it trainable
Flux.treelike(fmGAN)

"""
	fmGAN(generator, discriminator)

Basic fmGAN constructor.
"""
fmGAN(G::Flux.Chain, D::Flux.Chain; pz=randn) = fmGAN(G, freeze(G), D, freeze(D), 
	freeze(Flux.Chain(D.layers[1:end-1]...)), pz)

"""
	fmGAN(gsize, dsize, [pz, activation, layer])

Constructor for the fmGAN object. 
gsize - vector of Ints describing generator layers sizes
dsize - vector of Ints describing discriminator layers sizes, including the last scalar layer 
pz - code distribution
activation - activation function common to all layers
layer - layer type
"""
function fmGAN(gsize, dsize; pz = randn, activation = Flux.leakyrelu, layer = Flux.Dense)
	@assert size(gsize,1) >= 3
	@assert size(dsize,1) >= 3
	@assert dsize[end] == 1
	@assert gsize[end] == dsize[1]

	# generator
	generator = aelayerbuilder(gsize, activation, layer)

	# discriminator
	discriminator = discriminatorbuilder(dsize, activation, layer)

	return fmGAN(generator, discriminator, pz=pz)
end

################
### training ###
################

"""
	Dloss(fmgan, X, Z)

Discriminator loss.
"""
Dloss(fmgan::fmGAN, X, Z) = - Float(0.5)*(mean(log.(fmgan.d(X) + eps(Float))) + mean(log.(1 - fmgan.d(fmgan.gg(Z)) + eps(Float))))

"""
	Gloss(fmgan, Z)

Generator loss.
"""
Gloss(fmgan::fmGAN, Z) = - mean(log.(fmgan.dd(fmgan.g(Z)) + eps(Float)))

"""
	fmloss(fmgan, X, Z)

Feature matching loss computed on the penultimate discriminator layer.
"""
fmloss(fmgan::fmGAN, X, Z) = Flux.mse(fmgan.fmd(X), fmgan.fmd(fmgan.g(Z)))

"""
	rerr(fmgan, X, Z)

Crude estimate of reconstruction error.
"""
rerr(fmgan::fmGAN, X, Z) = Flux.mse(fmgan.g(Z), X) 

"""
	evalloss(fmgan, X, Z)
"""
evalloss(fmgan::fmGAN, X, Z) = print("discriminator loss: ", Flux.Tracker.data(Dloss(fmgan, X, Z)),  
	"\nfeature-matching loss: ", Flux.Tracker.data(fmloss(fmgan, X, Z)), 
	"\nreconstruction error: ", Flux.Tracker.data(rerr(fmgan, X, Z)), "\n\n")

"""
	fit!(fmgan, X, L, [alpha, iterations, cbit, verb, rdelta, eta])

Trains a fmGAN with the feature-matching loss.

fmgan - struct of type fmGAN
X - data array with instances as columns
L - batchsize
alpha - weight of the classical generator loss in the total loss
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
history - a dictionary for training progress control
eta - learning rate
"""
function fit!(fmgan::fmGAN, X, L; alpha = 1.0, iterations=1000, cbit = 200, verb = true, rdelta = Inf,
	history = nothing, eta = 0.001)
	# settings
	#Dopt = ADAM(params(fmgan.d))
	Dopt = ADAM(params(fmgan.d), eta)
	Gopt = ADAM(params(fmgan.g), eta)
	
	# problem size
	N = size(X,2)
	zdim = size(params(fmgan.g)[1],2)

	# train the fmGAN
	for i in 1:iterations
		# sample data and generate codes
		x = X[:,sample(1:N, L, replace=false)]
		z = getcode(fmgan, L)
                
        # discriminator training
        Dl = Dloss(fmgan, x, z)
    	if isnan(Dl)
			warn("Discriminator loss is NaN, ending fit.")
			return
		end    
		Flux.Tracker.back!(Dl)
        Dopt()
		
		# generator training	
        Gl = fmloss(fmgan, x, z) + Float(alpha)*Gloss(fmgan, z)
      	if isnan(Gl)
			warn("Generator loss is NaN, ending fit.")
			return
		end
		Flux.Tracker.back!(Gl)
        Gopt()
	
		# callback
		if verb && i%cbit==0
			evalloss(fmgan, x, z)
		end

		# save actual iteration data
		if history != nothing
			track!(fmgan, history, x, z)
		end

		# if a stopping condition is present
		if rdelta < Inf
			re = rerr(fmgan, x, z) 
			if re < rdelta
				println("Training ended prematurely after $i iterations,
					\nreconstruction error $re < $rdelta")
				break
			end
		end
	end
end

"""
	track!(fmgan, history, X, Z)

Save current progress of feature-matching fmGAN training.
"""
function track!(fmgan::fmGAN, history, X, Z)
	push!(history, :discriminator_loss, Flux.Tracker.data(Dloss(fmgan, X, Z)))
	push!(history, :generator_loss, Flux.Tracker.data(Gloss(fmgan, Z)))
	push!(history, :feature_matching_loss, Flux.Tracker.data(fmloss(fmgan, X, Z)))
	push!(history, :reconstruction_error, Flux.Tracker.data(rerr(fmgan, X, Z)))
end

############################
### auxilliary functions ###
############################
 
"""
	getcode(fmgan)

Generate a sample code from fmGAN.
"""
getcode(fmgan::fmGAN) = Float.(fmgan.pz(size(params(fmgan.g)[1],2)))

"""
	getcode(fmgan, N)

Generate codes from fmGAN.
"""
getcode(fmgan::fmGAN, n::Int) = Float.(fmgan.pz(size(params(fmgan.g)[1],2), n))

"""
	generate(fmgan)

Generate one sample.
"""
generate(fmgan::fmGAN) = fmgan.g(getcode(fmgan)).data

"""
	generate(fmgan, n)

Generate n samples.
"""
generate(fmgan::fmGAN, n::Int) = fmgan.g(getcode(fmgan, n)).data

"""
    discriminate(fmgan, X)

Return discrimiantor score.
"""
discriminate(fmgan::fmGAN, X) = Flux.Tracker.data(fmgan.d(X))

######################
### classification ###
######################

"""
	anomalyscore(fmgan, X, lambda)

Computes the anomaly score of X under given fmGAN using weighted average of reconstruction 
error and discriminator score.
"""
anomalyscore(fmgan::fmGAN, X::Array{Float, 1}, lambda) = 
	Flux.Tracker.data(Float(1 - lambda)*-mean(log.(fmgan.d(X) + eps(Float))) + 
	Float(lambda)*rerr(fmgan, X, getcode(fmgan, size(X,2))))
anomalyscore(fmgan::fmGAN, X::Array{Float, 2}, lambda) = 
	reshape(mapslices(y -> anomalyscore(fmgan, y, lambda), X, 1), size(X,2))

"""
	classify(fmgan, x, threshold, lambda)

Classify an instance x using the discriminator and error losses and a threshold in a 
feature-matching GAN setting.
"""
classify(fmgan::fmGAN, X, threshold, lambda) = Int.(anomalyscore(fmgan, X, lambda) .> Float(threshold))

"""
	getthreshold(fmgan, x, contamination, lambda, [Beta])

Compute threshold for fmGAN classification based on known contamination level.
"""
function getthreshold(fmgan::fmGAN, X, contamination, lambda; Beta = 1.0)
	N = size(X, 2)
	Beta = Float(Beta)
	# get anomaly score
	ascore = anomalyscore(fmgan, X, lambda)
	# sort it
	ascore = sort(ascore)
	aN = Int(ceil(N*contamination)) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	(aN > 0)? (return Beta*ascore[end-aN] + (1-Beta)*ascore[end-aN+1]) : (return ascore[end])
end

################################################################################
### A SK-learn like model based on fmGAN with the same methods and some new. ###
################################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct fmGANmodel <:genmodel
	fmgan::fmGAN
	lambda::Real
	threshold::Real
	contamination::Real
	L::Int
	iterations::Int
	cbit::Real
	verbfit::Bool
	rdelta
	alpha
	Beta
	history
	eta::Real
end

"""
	fmGANmodel(gsize, dsize, lambda, threshold, contamination, L, iterations, 
	cbit, verbfit, [pz, activation, rdelta, alpha, Beta, tracked, eta])

Initialize a generative adversarial net model for classification with given parameters.

gsize - generator architecture
dsize - discriminator architecture
lambda - weighs between the reconstruction error (1) and discriminator score (0) in classification
threshold - anomaly score threshold for classification, is set automatically using contamination during fit
contamination - percentage of anomalous samples in all data for automatic threshold computation
L - batchsize
iterations - number of training iterations
cbit - current training progress is printed every cbit iterations
verbfit - is progress printed?
pz [randn] - code generating distribution
activation [Flux.relu] - activation function
rdelta [Inf] - training stops if reconstruction error is smaller than rdelta
alpha [1.0] - weight of the classical generator loss -D(G(Z)) in the total generator loss
Beta [Beta] - how tight around normal data is the automatically computed threshold
tracked [false] - is training progress (losses) stored?
eta [0.001] - learning rate
"""
function fmGANmodel(gsize::Array{Int64,1}, dsize::Array{Int64,1},
	lambda::Real, threshold::Real, contamination::Real, L::Int, iterations::Int, 
	cbit::Int, verbfit::Bool; pz = randn, activation = Flux.leakyrelu, 
	layer = Flux.Dense, rdelta = Inf,
	alpha = 1.0, Beta = 1.0, tracked = false, eta= 0.001)
	# construct the fmGAN object
	fmgan = fmGAN(gsize, dsize, pz = pz, activation = activation, layer = layer)
	(tracked)? history = MVHistory() : history = nothing
	model = fmGANmodel(fmgan, lambda, threshold, contamination, L, iterations, cbit, 
		verbfit, rdelta, alpha, Beta, history, eta)
	return model
end

# reimplement some methods of fmGAN
Dloss(model::fmGANmodel, X, Z) = Dloss(model.fmgan, X, Z)
Gloss(model::fmGANmodel, Z) = Gloss(model.fmgan, Z)
fmloss(model::fmGANmodel, X, Z) = fmloss(model.fmgan, X, Z)
rerr(model::fmGANmodel, X, Z) = rerr(model.fmgan, X, Z)
evalloss(model::fmGANmodel, X, Z) = evalloss(model.fmgan, X, Z) 
generate(model::fmGANmodel) = generate(model.fmgan)
generate(model::fmGANmodel, n::Int) = generate(model.fmgan, n)
anomalyscore(model::fmGANmodel, X) = anomalyscore(model.fmgan, X, model.lambda)
classify(model::fmGANmodel, x) = classify(model.fmgan, x, model.threshold, model.lambda)
getthreshold(model::fmGANmodel, X) = getthreshold(model.fmgan, X, model.contamination, model.lambda, Beta = model.Beta)
getcode(model::fmGANmodel) = getcode(model.fmgan)
getcode(model::fmGANmodel, n) = getcode(model.fmgan, n)
discriminate(model::fmGANmodel, X) = discriminate(model.fmgan, X)
params(model::fmGANmodel) = Flux.params(model.fmgan)

"""
	setthreshold!(model, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::fmGANmodel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model, X)

Trains a fmGANmodel.
"""
function fit!(model::fmGANmodel, X)
	# train the fmGAN NN
	fit!(model.fmgan, X, model.L; alpha = model.alpha, iterations=model.iterations, 
	cbit = model.cbit, verb = model.verbfit, rdelta = model.rdelta,
	history = model.history, eta = model.eta)
end

"""
	predict(model, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::fmGANmodel, X) 
	return classify(model, X)
end
