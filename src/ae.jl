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
Flux.@treelike AE

"""
	AE(esize, dsize; [activation, layer])

Initialize an autoencoder with given encoder size and decoder size.

esize - vector of ints specifying the width anf number of layers of the encoder
\ndsize - size of decoder
\nactivation [Flux.relu] - arbitrary activation function
\nlayer [Flux.Dense] - layer type
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
evalloss(ae::AE, X) = println("loss: ", getlosses(ae,X)[1], "\n")

"""
	getlosses(ae, X)

Return the numeric values of current losses.
"""
getlosses(ae::AE, X) = (
	Flux.Tracker.data(loss(ae, X))
	)

"""
	fit!(ae, X, batchsize, [iterations, cbit, nepochs, verb, rdelta, history, eta])

Trains the AE.

ae - AE type object
\nX - data array with instances as columns
\nbatchsize - batchsize
\niterations [1000] - number of iterations
\ncbit [200] - after this # of iterations, output is printed
\nnepochs [nothing] - if this is supplied, epoch training will be used instead of fixed iterations
\nverb [true] - if output should be produced
\nrdelta [Inf] - stopping condition for reconstruction error
\nhistory [nothing] - MVHistory() to be filled with data of individual iterations
\neta [0.001] - learning rate
"""
function fit!(ae::AE, X, batchsize; iterations=1000, cbit = 200, nepochs = nothing, 
	verb = true, rdelta = Inf, history = nothing, eta = 0.001)
	# optimizer
	opt = ADAM(params(ae), eta)

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
		_l = getlosses(ae, x)
	end

	# training
	for (i,x) in enumerate(sampler)
		# gradient computation and update
		l = loss(ae, x)
		Flux.Tracker.back!(l)
		opt()

		# progress
		if verb 
			if (i%cbit == 0 || i == 1)
				_l = getlosses(ae, x)
			end
			ProgressMeter.next!(p; showvalues = [(:loss,_l)])
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
	reshape(mapslices(y -> anomalyscore(ae, y), X, dims = 1), size(X,2))
anomalyscore(ae::AE, X::Union{Array{T, 1},Array{T, 2}} where T<:Real) = 
	anomalyscore(ae,Float.(X))

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
	nepochs
	verbfit::Bool
	rdelta::Real
	Beta::Float
	history
	eta::Real
end

"""
	AEmodel(esize, dsize, [batchsize, threshold, contamination, iteration, cbit, 
	nepochs, activation, rdelta, Beta, tracked, eta])

Initialize an autoencoder model with given parameters.

esize - encoder architecture
\ndsize - decoder architecture
\nbatchsize [256] - batchsize
\nthreshold [0.0] - anomaly score threshold for classification, is set automatically using contamination during fit
\ncontamination [0.0] - percentage of anomalous samples in all data for automatic threshold computation
\niterations [10000] - number of training iterations
\ncbit [1000] - current training progress is printed every cbit iterations
\nnepochs [nothing] - if this is supplied, epoch training will be used instead of fixed iterations
\nverbfit [true] - is progress printed?
\nactivation [Flux.relu] - activation function
\nrdelta [Inf] - training stops if reconstruction error is smaller than rdelta
\nBeta [1.0] - how tight around normal data is the automatically computed threshold
\ntracked [false] - is training progress (losses) stored?
\neta [0.001] - learning rate
"""
function AEmodel(esize::Array{Int64,1}, dsize::Array{Int64,1};
	batchsize::Int=256, threshold::Real=0.0, contamination::Real=0.0, 
	iterations::Int=10000, cbit::Real=1000,
	nepochs = nothing, verbfit::Bool=true, 
	activation = Flux.relu, rdelta = Inf, Beta = 1.0,
	tracked = false, layer = Flux.Dense, eta = 0.001)
	# construct the AE object
	ae = AE(esize, dsize, activation = activation, layer = layer)
	tracked ? history = MVHistory() : history = nothing
	model = AEmodel(ae, batchsize, threshold, contamination, iterations, cbit, 
		nepochs, verbfit, rdelta, Beta, history, eta)
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
	cbit = model.cbit, nepochs = model.nepochs,
	verb = model.verbfit, rdelta = model.rdelta,
	history = model.history, eta = model.eta)
end

"""
	predict(model::AEmodel, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::AEmodel, X)	
	return classify(model, X)
end
