using Flux
import Base.Iterators.repeated

### ae NN construction ###

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
	AE(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int; 
	activation = Flux.relu)

Initialize an autoencoder with given parameters.
"""
function AE(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int;
	activation = Flux.relu)
	# construct the encoder
	encoder = Dense(indim,hiddendim,activation)
	for i in 2:nlayers
	    encoder = Chain(encoder, Dense(hiddendim,hiddendim,activation))
	end
	encoder = Chain(encoder, Dense(hiddendim, latentdim))
	    
	# construct the decoder
	decoder = Dense(latentdim, hiddendim, Flux.relu)
	for i in 2:nlayers
	    decoder = Chain(decoder, Dense(hiddendim, hiddendim, Flux.relu))
	end
	decoder = Chain(decoder, Dense(hiddendim, indim))    

	# finally construct the ae struct
	ae = AE(encoder, decoder)

	return ae
end

### fitting ###

"""
	loss(ae::AE, X, Y)

Reconstruction error.
"""
loss(ae::AE, X) = Flux.mse(ae(X), X)

"""
	evalcb(ae::AE, X)

Print ae loss function values.	
"""
evalloss(ae::AE, X) = print("loss: ", loss(ae, X).data[1])

"""
	fit!(ae::AE, X)

Trains the AE.
"""
function fit!(ae::AE, X; iterations=1000, throttle = 5, verb = true)
	# settings
	opt = ADAM(params(ae))
	if iterations != 0 # if X are complete data that should be trained on numerous times
		dataset = repeated((ae, X), iterations) # Y=x
		evalcb = () -> print("loss: ", loss(ae, X).data[1], "\n\n")	
	else
		dataset = X # if x is already an iterable to be trained on
		evalcb = () -> print("loss: ", loss(ae, dataset[1][2]).data[1], "\n\n")	
	end
	
	cb = Flux.throttle(evalcb, throttle)

	# train
	if verb
		Flux.train!(loss, dataset, opt, cb = cb)
	else
		Flux.train!(loss, dataset, opt)
	end
end

### ae output ###

"""
	classify(ae::AE, x, threshold)

Classify an instance x using reconstruction error and threshold.
"""
classify(ae::AE, x, threshold) = (loss(ae, x) > threshold)? 1 : 0
classify(ae::AE, x::Array{Float64,1}, threshold) = (loss(ae, x) > threshold)? 1 : 0
classify(ae::AE, X::Array{Float64,2}, threshold) = reshape(mapslices(y -> classify(ae, y, threshold), X, 1), size(X,2))

"""
	get_threshold(ae::AE, x, contamination)

Compute threshold for AE classification based on known contamination level.
"""
function get_threshold(ae::AE, x, contamination)
	N = size(x, 2)
	# get reconstruction errors
	xerr  = mapslices(y -> loss(ae, y), x, 1)
	# create ordinary array from the tracked array
	xerr = reshape([e.data[1] for e in xerr], N)
	# sort it
	xerr = sort(xerr)
	aN = Int(floor(N*contamination)) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	return (xerr[end-aN]+xerr[end-aN+1])/2
end

### A SK-learn like model based on AE with the same methods and some new. ###
""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct AEmodel
	ae::AE
	threshold::Real
	contamination::Real
	iterations::Int
	cbthrottle::Real
	verbfit::Bool
end

"""
	AEmodel(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int,
	threshold::Real, contamination::Real, iteration::Int, cbthrottle::Real)

Initialize a variational autoencoder model with given parameters.
"""
function AEmodel(indim::Int, hiddendim::Int, latentdim::Int, nlayers::Int,
	activation,	threshold::Real, contamination::Real, iterations::Int, 
	cbthrottle::Real, verbfit::Bool)
	# construct the AE object
	ae = AE(indim, hiddendim, latentdim, nlayers, activation = activation)
	model = AEmodel(ae, threshold, contamination, iterations, cbthrottle, verbfit)
	return model
end

# reimplement some methods of AE
(model::AEmodel)(x) = model.ae(x)   
loss(model::AEmodel, X) = loss(model.ae, X)
evalloss(model::AEmodel, X) = evalloss(model.ae, X)
classify(model::AEmodel, x) = classify(model.ae, x, model.threshold)
classify(model::AEmodel, x, threshold) = classify(model.ae, x, threshold)
get_threshold(model::AEmodel, x) = get_threshold(model.ae, x, model.contamination)
get_threshold(model::AEmodel, x, contamination) = get_threshold(model.ae, x, contamination)

"""
	fit!(model::AEmodel, X)

Fit the AE model, instances are columns of X.	
"""
fit!(model::AEmodel, X) = fit!(model.ae, X, iterations = model.iterations, 
	throttle = model.cbthrottle, verb = model.verbfit)

"""
	predict(model::AEmodel, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::AEmodel, X)
	model.threshold = get_threshold(model.ae, X, model.contamination)
	return classify(model.ae, X, model.threshold)
end