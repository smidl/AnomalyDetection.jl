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
	gg # non-trainable generator copy
	d # discriminator
	dd # non-trainable discriminator copy
	pz # code distribution
end


"""
	fmGAN(generator, discriminator)

Basic fmGAN constructor.
"""
fmGAN(G::Flux.Chain, D::Flux.Chain; pz=randn) = fmGAN(G, freeze(G), D, freeze(D), pz)

"""
	fmGAN(gsize, dsize, [pz, activation])

Constructor for the fmGAN object. 
gsize - vector of Ints describing generator layers sizes
dsize - vector of Ints describing discriminator layers sizes, including the last scalar layer 
pz - code distribution
activation - activation function common to all layers
"""
function fmGAN(gsize, dsize; pz = randn, activation = Flux.leakyrelu)
	@assert size(gsize,1) >= 3
	@assert size(dsize,1) >= 3
	@assert dsize[end] == 1
	@assert gsize[end] == dsize[1]

	# generator
	generator = Dense(gsize[1], gsize[2], activation)
	for i in 3:(size(gsize,1)-1)
		generator = Chain(generator, Dense(gsize[i-1], gsize[i], activation))
	end
	generator = Chain(generator, Dense(gsize[end-1], gsize[end]))

	# discriminator
	discriminator = Dense(dsize[1], dsize[2], activation)
	for i in 3:(size(dsize,1)-1)
		discriminator = Chain(discriminator, Dense(dsize[i-1], dsize[i], activation))
	end
	discriminator = Chain(discriminator, Dense(dsize[end-1], dsize[end], σ))

	return fmGAN(generator, discriminator, pz=pz)
end

################
### training ###
################

"""
	Dloss(fmgan, X, Z)

Discriminator loss.
"""
Dloss(fmgan::fmGAN, X, Z) = - 0.5*(mean(log.(fmgan.d(X))) + mean(log.(1 - fmgan.d(fmgan.gg(Z)))))

"""
	Gloss(fmgan, Z)

Generator loss.
"""
Gloss(fmgan::fmGAN, Z) = - mean(log.(fmgan.dd(fmgan.g(Z))))

"""
	fmloss(fmgan, X, Z)

Feature matching loss computed on the penultimate discriminator layer.
"""
fmloss(fmgan::fmGAN, X, Z) = Flux.mse(fmgan.dd.layers[1](X), fmgan.dd.layers[1](fmgan.g(Z)))

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
	fit!(fmgan, X, L, [alpha, iterations, cbit, verb, rdelta])

Trains a fmGAN with the feature-matching loss.

fmgan - struct of type fmGAN
X - data array with instances as columns
L - batchsize
alpha - weight of the classical generator loss in the total loss
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
traindata - a dictionary for training progress control
"""
function fit!(fmgan::fmGAN, X, L; alpha = 1.0, iterations=1000, cbit = 200, verb = true, rdelta = Inf,
	traindata = nothing)
	# settings
	#Dopt = ADAM(params(fmgan.d))
	Dopt = SGD(params(fmgan.d))
	Gopt = ADAM(params(fmgan.g))
	
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
        Flux.Tracker.back!(Dl)
        Dopt()
		
		# generator training	
        Gl = fmloss(fmgan, x, z) + alpha*Gloss(fmgan, z)
        Flux.Tracker.back!(Gl)
        Gopt()
	
		# callback
		if verb && i%cbit==0
			evalloss(fmgan, x, z)
		end

		# save actual iteration data
		if traindata != nothing
			track!(fmgan, traindata, x, z)
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
	track!(fmgan, traindata, X, Z)

Save current progress of feature-matching fmGAN training.
"""
function track!(fmgan::fmGAN, traindata, X, Z)
	# Dloss
	if haskey(traindata, "discriminator loss")
		push!(traindata["discriminator loss"], Flux.Tracker.data(Dloss(fmgan, X, Z)))
	else
		traindata["discriminator loss"] = [Flux.Tracker.data(Dloss(fmgan, X, Z))]
	end

	# Gloss
	if haskey(traindata, "generator loss")
		push!(traindata["generator loss"], Flux.Tracker.data(Gloss(fmgan, Z)))
	else
		traindata["generator loss"] = [Flux.Tracker.data(Gloss(fmgan, Z))]
	end

	# feature matching loss
	if haskey(traindata, "feature-matching loss")
		push!(traindata["feature-matching loss"], Flux.Tracker.data(fmloss(fmgan, X, Z)))
	else
		traindata["feature-matching loss"] = [Flux.Tracker.data(fmloss(fmgan, X, Z))]
	end

	# reconstruction error
	if haskey(traindata, "reconstruction error")
		push!(traindata["reconstruction error"], Flux.Tracker.data(rerr(fmgan, X, Z)))
	else
		traindata["reconstruction error"] = [Flux.Tracker.data(rerr(fmgan, X, Z))]
	end
end

############################
### auxilliary functions ###
############################
 
"""
	getcode(fmgan)

Generate a sample code from fmGAN.
"""
getcode(fmgan::fmGAN) = fmgan.pz(size(params(fmgan.g)[1],2))

"""
	getcode(fmgan, N)

Generate codes from fmGAN.
"""
getcode(fmgan::fmGAN, n::Int) = fmgan.pz(size(params(fmgan.g)[1],2), n)

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
anomalyscore(fmgan::fmGAN, X, lambda) = (1 - lambda)*-Flux.Tracker.data(mean(log.(fmgan.d(X)))) + lambda*rerr(fmgan, X, getcode(fmgan, size(X,2)))

"""
	classify(fmgan, x, threshold, lambda)

Classify an instance x using the discriminator and error losses and a threshold in a 
feature-matching GAN setting.
"""
classify(fmgan::fmGAN, x, threshold, lambda) = (anomalyscore(fmgan, x, lambda) > threshold)? 1 : 0
classify(fmgan::fmGAN, x::Array{Float64,1}, threshold, lambda) = (anomalyscore(fmgan, x, lambda) > threshold)? 1 : 0
classify(fmgan::fmGAN, X::Array{Float64,2}, threshold, lambda) = reshape(mapslices(y -> classify(fmgan, y, threshold, lambda), X, 1), size(X,2))

"""
	getthreshold(fmgan, x, contamination, lambda, [Beta])

Compute threshold for fmGAN classification based on known contamination level.
"""
function getthreshold(fmgan::fmGAN, X, contamination, lambda; Beta = 1.0)
	N = size(X, 2)
	# get anomaly score
	ascore = mapslices(y -> anomalyscore(fmgan, y, lambda), X, 1)
	# create ordinary array from the tracked array
	ascore = reshape([Flux.Tracker.data(s)[1] for s in ascore], N)
	# sort it
	ascore = sort(ascore)
	aN = max(Int(floor(N*contamination)),1) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	return Beta*ascore[end-aN] + (1-Beta)ascore[end-aN+1]
end

################################################################################
### A SK-learn like model based on fmGAN with the same methods and some new. ###
################################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct fmGANmodel
	fmgan::fmGAN
	lambda::Real
	threshold::Real
	contamination::Real
	L::Int
	iterations::Int
	cbit::Real
	verbfit::Bool
	rdelta::Float64
	alpha::Float64
	Beta::Float64
	traindata
end

"""
	fmGANmodel(gsize, dsize, lambda, threshold, contamination, L, iterations, 
	cbit, verbfit, [pz, activation, rdelta, alpha, Beta, tracked])

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
"""
function fmGANmodel(gsize::Array{Int64,1}, dsize::Array{Int64,1},
	lambda::Real, threshold::Real, contamination::Real, L::Int, iterations::Int, 
	cbit::Int, verbfit::Bool; pz = randn, activation = Flux.leakyrelu, rdelta = Inf,
	alpha = 1.0, Beta = 1.0, tracked = false)
	# construct the fmGAN object
	fmgan = fmGAN(gsize, dsize, pz = pz, activation = activation)
	(tracked)? traindata = Dict{Any, Any}() : traindata = nothing
	model = fmGANmodel(fmgan, lambda, threshold, contamination, L, iterations, cbit, 
		verbfit, rdelta, alpha, Beta, traindata)
	return model
end

# reimplement some methods of fmGAN
Dloss(model::fmGANmodel, X, Z) = Dloss(model.fmgan, X, Z)
fmloss(model::fmGANmodel, X, Z) = fmloss(model.fmgan, X, Z)
rerr(model::fmGANmodel, X, Z) = rerr(model.fmgan, X, Z)
evalloss(model::fmGANmodel, X, Z) = evalloss(model.fmgan, X, Z) 
generate(model::fmGANmodel) = generate(model.fmgan)
generate(model::fmGANmodel, n::Int) = generate(model.fmgan, n)
anomalyscore(model::fmGANmodel, X) = anomalyscore(model.fmgan, X, model.lambda)
classify(model::fmGANmodel, x) = classify(model.fmgan, x, model.threshold, model.lambda)
classify(model::fmGANmodel, x::Array{Float64,1}) = classify(model.fmgan, x, model.threshold, model.lambda)
classify(model::fmGANmodel, X::Array{Float64,2}) = classify(model.fmgan, X, model.threshold, model.lambda)
getthreshold(model::fmGANmodel, X) = getthreshold(model.fmgan, X, model.contamination, model.lambda, Beta = model.Beta)
getcode(model::fmGANmodel) = getcode(model.fmgan)
getcode(model::fmGANmodel, n) = getcode(model.fmgan, n)
discriminate(model::fmGANmodel, X) = discriminate(model.fmgan, X)

"""
	plot(model)

Plot the model losses.
"""
function plot(model::fmGANmodel)
	# plot model loss
	if model.traindata == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
	    figure()
	    title("model loss")
	    if model.alpha>0.0
		    y1, = plot(model.traindata["generator loss"], label = "generator loss", c = "b")
    	    ylabel("Gloss + Dloss")
		else
		   ylabel("Dloss")
		end
	    y2, = plot(model.traindata["discriminator loss"], label = "discriminator loss", c = "g")

	    xlabel("iteration")
	    ax = gca()
	    
	    ax2 = ax[:twinx]()
	    y3, = plot(model.traindata["feature-matching loss"], label = "feature-matching loss", c = "m")
	    y4, = plot(model.traindata["reconstruction error"], label = "reconstruction error", c = "r")
	    ylabel("FMloss + reconstruction error")
	    if model.alpha>0.0
		    legend([y1, y2, y3, y4], ["generator loss",   "discriminator loss",
		    	"feature-matching loss", "reconstruction error"])
		else
			legend([y2, y3, y4], ["discriminator loss", 
	    	"feature-matching loss", "reconstruction error"])
		end
	    show()
	end
end

"""
	setthreshold!(model, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::fmGANmodel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model, X, Y)

Trains a fmGANmodel.
"""
function fit!(model::fmGANmodel, X, Y)
	# train the NN only on normal samples
	nX = X[:, Y.==0]

	# train the fmGAN NN
	fit!(model.fmgan, nX, model.L; alpha = model.alpha, iterations=model.iterations, 
	cbit = model.cbit, verb = model.verbfit, rdelta = model.rdelta,
	traindata = model.traindata)

	# now set the threshold using contamination rate
	model.contamination = size(Y[Y.==1],1)/size(Y[Y.==0],1)
	setthreshold!(model, X)
end

"""
	predict(model, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::fmGANmodel, X) 
	return classify(model, X)
end
