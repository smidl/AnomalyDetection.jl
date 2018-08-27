##########################
### ae NN construction ###
##########################

"""
	AE{encoder, sampler, decoder}

Flux-like structure for the basic autoencoder.
"""
struct AE{E, D}
	encoder::E
	decoder::D
end

# make the struct callable
(ae::AE)(X) = ae.decoder(ae.encoder(X))

# and make it trainable
Flux.treelike(AE)

"""
	AE(esize, dsize; [activation])

Initialize an autoencoder with given encoder size and decoder size.
esize - vector of ints specifying the width anf number of layers of the encoder
dsize - size of decoder
activation - arbitrary activation function
"""
function AE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu,
		layer = Flux.Dense)
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == dsize[1] 
	@assert esize[1] == dsize[end]

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	ae = AE(encoder, decoder)

	return ae
end

################
### training ###
################

"""
	loss(ae, X)

Reconstruction error.
"""
loss(ae::AE, X) = Flux.mse(ae(X), X)

"""
	evalloss(ae, X)

Print ae loss function values.	
"""
evalloss(ae::AE, X) = println("loss: ", Flux.Tracker.data(loss(ae, X)), "\n")

"""
	fit!(ae, X, batchsize, [iterations, cbit, verb, rdelta, tracked, eta])

Trains the AE.
ae - AE type object
X - data array with instances as columns
batchsize - batchsize
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
history - MVHistory() to be filled with data of individual iterations
eta - learning rate
"""
function fit!(ae::AE, X, batchsize; iterations=1000, cbit = 200, verb = true, rdelta = Inf, 
	history = nothing, eta = 0.001)
	# optimizer
	opt = ADAM(params(ae), eta)

	# training
	for i in 1:iterations
		# sample from data
		x = X[:, sample(1:size(X,2), batchsize, replace = false)]

		# gradient computation and update
		l = loss(ae, x)
		Flux.Tracker.back!(l)
		opt()

		# callback
		if verb && i%cbit == 0
			evalloss(ae, x)
		end

		# save actual iteration data
		if history != nothing
			track!(ae, history, x)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = Flux.Tracker.data(l)[1]
			if re < rdelta
				if verb
					println("Training ended prematurely after $i iterations,\n",
						"reconstruction error $re < $rdelta")
				end
				break
			end
		end
	end	
end

"""
	track!(ae, history, X)

Save current progress.
"""
function track!(ae::AE, history::MVHistory, X)
	push!(history, :loss, Flux.Tracker.data(loss(ae,X)))
end

#################
### ae output ###
#################

"""
	anomalyscore(ae, X)

Compute anomaly score for X.
"""
anomalyscore(ae::AE, X::Array{Float, 1}) = Flux.Tracker.data(loss(ae, X))
anomalyscore(ae::AE, X::Array{Float, 2}) = 
	reshape(mapslices(y -> anomalyscore(ae, y), X, 1), size(X,2))

"""
	classify(ae, x, threshold)

Classify an instance x using reconstruction error and threshold.
"""
classify(ae::AE, X, threshold) = Int.(anomalyscore(ae, X) .> threshold)

#############################################################################
### A SK-learn like model based on AE with the same methods and some new. ###
#############################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct AEmodel <: genmodel
	ae::AE
	batchsize::Int
	threshold::Real
	contamination::Real
	iterations::Int
	cbit::Real
	verbfit::Bool
	rdelta::Real
	Beta::Float
	history
	eta::Real
end

"""
	AEmodel(esize, dsize, [batchsize, threshold, contamination, iteration, cbit, 
	activation, rdelta, Beta, tracked, eta])

Initialize an autoencoder model with given parameters.

esize - encoder architecture
dsize - decoder architecture
batchsize - batchsize
threshold - anomaly score threshold for classification, is set automatically using contamination during fit
contamination - percentage of anomalous samples in all data for automatic threshold computation
iterations - number of training iterations
cbit - current training progress is printed every cbit iterations
verbfit - is progress printed?
activation [Flux.relu] - activation function
rdelta [Inf] - training stops if reconstruction error is smaller than rdelta
Beta [1.0] - how tight around normal data is the automatically computed threshold
tracked [false] - is training progress (losses) stored?
eta - learning rate
"""
function AEmodel(esize::Array{Int64,1}, dsize::Array{Int64,1};
	batchsize::Int=1, threshold::Real=0.0, contamination::Real=0.0, 
	iterations::Int=10000, cbit::Real=1000, verbfit::Bool=true, 
	activation = Flux.relu, rdelta = Inf, Beta = 1.0,
	tracked = false, layer = Flux.Dense, eta = 0.001)
	# construct the AE object
	ae = AE(esize, dsize, activation = activation, layer = layer)
	(tracked)? history = MVHistory() : history = nothing
	model = AEmodel(ae, batchsize, threshold, contamination, iterations, cbit, verbfit, rdelta, 
		Beta, history, eta)
	return model
end

# reimplement some methods of AE
(model::AEmodel)(x) = model.ae(x)   
loss(model::AEmodel, X) = loss(model.ae, X)
evalloss(model::AEmodel, X) = evalloss(model.ae, X)
anomalyscore(model::AEmodel, X) = anomalyscore(model.ae, X)
classify(model::AEmodel, x) = classify(model.ae, x, model.threshold)
getthreshold(model::AEmodel, x) = getthreshold(model.ae, x, model.contamination, Beta = model.Beta)
params(model::AEmodel) = Flux.params(model.ae)

"""
	setthreshold!(model::AEmodel, X)

Set model classification threshold based ratior of labels in Y.
"""
function setthreshold!(model::AEmodel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model::AEmodel, X)

Fit the AE model, instances are columns of X, X are normal samples!!!.	
"""
function fit!(model::AEmodel, X) 
	# train
	fit!(model.ae, X, model.batchsize, iterations = model.iterations, 
	cbit = model.cbit, verb = model.verbfit, rdelta = model.rdelta,
	history = model.history, eta = model.eta)
end

"""
	predict(model::AEmodel, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::AEmodel, X)	
	return classify(model, X)
end
