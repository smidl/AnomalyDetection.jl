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
function AE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu)
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == dsize[1] 
	@assert esize[1] == dsize[end]

	# construct the encoder
	encoder = Dense(esize[1],esize[2],activation)
	for i in 3:(size(esize,1)-1)
	    encoder = Chain(encoder, Dense(esize[i-1],esize[i],activation))
	end
	encoder = Chain(encoder, Dense(esize[end-1], esize[end]))
	    
	# construct the decoder
	decoder = Dense(dsize[1],dsize[2],activation)
	for i in 3:(size(dsize,1)-1)
	    decoder = Chain(decoder, Dense(dsize[i-1],dsize[i],activation))
	end
	decoder = Chain(decoder, Dense(dsize[end-1], dsize[end]))
	
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
evalloss(ae::AE, X) = println("loss: ", loss(ae, X).tracker.data, "\n")

"""
	fit!(ae, X, [iterations, cbit, verb, rdelta, tracked])

Trains the AE.
ae - AE type object
X - data array with instances as columns
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
traindata - dict to be filled with data of individual iterations
"""
function fit!(ae::AE, X; iterations=1000, cbit = 200, verb = true, rdelta = Inf, traindata = nothing)
	# optimizer
	opt = ADAM(params(ae))

	# training
	for i in 1:iterations
		# gradient computation and update
		l = loss(ae, X)
		Flux.Tracker.back!(l)
		opt()

		# callback
		if verb && i%cbit == 0
			evalloss(ae, X)
		end

		# save actual iteration data
		if traindata != nothing
			track!(ae, traindata, X)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = loss(ae, X).tracker.data
			if re < rdelta
				println("Training ended prematurely after $i iterations,\n",
					"reconstruction error $re < $rdelta")
				break
			end
		end
	end	
end

"""
	track!(ae, traindata, X)

Save current progress.
"""
function track!(ae::AE, traindata, X)
	if haskey(traindata, "loss")
		push!(traindata["loss"], Flux.Tracker.data(loss(ae,X)))
	else
		traindata["loss"] = [Flux.Tracker.data(loss(ae, X))]
	end
end

#################
### ae output ###
#################

"""
	anomalyscore(ae, X)

Compute anomaly score for X.
"""
anomalyscore(ae::AE, X) = loss(ae, X)

"""
	classify(ae, x, threshold)

Classify an instance x using reconstruction error and threshold.
"""
classify(ae::AE, x, threshold) = (anomalyscore(ae, x) > threshold)? 1 : 0
classify(ae::AE, x::Array{Float64,1}, threshold) = (anomalyscore(ae, x) > threshold)? 1 : 0
classify(ae::AE, X::Array{Float64,2}, threshold) = reshape(mapslices(y -> classify(ae, y, threshold), X, 1), size(X,2))

"""
	getthreshold(ae, x, contamination, [Beta])

Compute threshold for AE classification based on known contamination level.
"""
function getthreshold(ae::AE, x, contamination; Beta = 1.0)
	N = size(x, 2)
	# get reconstruction errors
	xerr  = mapslices(y -> loss(ae, y), x, 1)
	# create ordinary array from the tracked array
	xerr = reshape([e.tracker.data for e in xerr], N)
	# sort it
	xerr = sort(xerr)
	aN = max(Int(floor(N*contamination)),1) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	return Beta*xerr[end-aN] + (1-Beta)xerr[end-aN+1]
end

#############################################################################
### A SK-learn like model based on AE with the same methods and some new. ###
#############################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct AEmodel
	ae::AE
	threshold::Real
	contamination::Real
	iterations::Int
	cbit::Real
	verbfit::Bool
	rdelta::Real
	Beta::Float64
	traindata
end

"""
	AEmodel(esize, dsize, threshold, contamination, iteration, cbit, 
	[activation, rdelta, Beta, tracked])

Initialize an autoencoder model with given parameters.
"""
function AEmodel(esize::Array{Int64,1}, dsize::Array{Int64,1},
	threshold::Real, contamination::Real, iterations::Int, 
	cbit::Real, verbfit::Bool; activation = Flux.relu, rdelta = Inf, Beta = 1.0,
	tracked = false)
	# construct the AE object
	ae = AE(esize, dsize, activation = activation)
	(tracked)? traindata = Dict{Any, Any}() : traindata = nothing
	model = AEmodel(ae, threshold, contamination, iterations, cbit, verbfit, rdelta, 
		Beta, traindata)
	return model
end

# reimplement some methods of AE
(model::AEmodel)(x) = model.ae(x)   
loss(model::AEmodel, X) = loss(model.ae, X)
evalloss(model::AEmodel, X) = evalloss(model.ae, X)
classify(model::AEmodel, x) = classify(model.ae, x, model.threshold)
getthreshold(model::AEmodel, x) = getthreshold(model.ae, x, model.contamination, Beta = model.Beta)

"""
	plot(model)

Plot the model loss.
"""
function plot(model::AEmodel)
	# plot model loss
	if model.traindata == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
	    figure()
	    title("model loss")
	    plot(model.traindata["loss"])
	    xlabel("iteration")
	    ylabel("loss")
	    show()
	end
end

"""
	setthreshold!(model::AEmodel, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::AEmodel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model::AEmodel, X, Y)

Fit the AE model, instances are columns of X.	
"""
function fit!(model::AEmodel, X, Y) 
	# train the NN only on normal samples
	nX = X[:, Y.==0]

	# train
	fit!(model.ae, nX, iterations = model.iterations, 
	cbit = model.cbit, verb = model.verbfit, rdelta = model.rdelta,
	traindata = model.traindata)

	# now set the threshold using contamination rate
	model.contamination = size(Y[Y.==1],1)/size(Y[Y.==0],1)
	setthreshold!(model, X)
end

"""
	predict(model::AEmodel, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::AEmodel, X)	
	return classify(model, X)
end
