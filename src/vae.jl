###########################
### vae NN construction ###
###########################

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
	mu(vae, X)

Extract means from the last encoder layer.
"""
mu(vae::VAE, X) = X[1:Int(size(vae.encoder.layers[end].W,1)/2),:]

"""
	sigma(vae, X)

Extract sigmas from the last encoder layer.
"""
sigma(vae::VAE, X) = softplus(X[Int(size(vae.encoder.layers[end].W,1)/2+1):end,:]) + 1e-6

"""
	sample_z(vae, X)

Sample from the last encoder layer.
"""
function sample_z(vae::VAE, X)
	res = mu(vae, X)
	return res + randn(size(res)) .* sigma(vae,X)
end

"""
	VAE(esize, dsize; [activation])

Initialize a variational autoencoder with given encoder size and decoder size.
esize - vector of ints specifying the width anf number of layers of the encoder
dsize - size of decoder
activation - arbitrary activation function
"""
function VAE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu)
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == 2*dsize[1] 
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
	vae = VAE(encoder, sample_z, decoder)

	return vae
end

################
### training ###
################

"""
	KL(vae, X)

KL divergence between the encoder parameters and unit gaussian.
"""
KL(vae::VAE, X) = 1/2*mean(sum(sigma(vae, vae.encoder(X)).^2 + mu(vae, vae.encoder(X)).^2 - log.(sigma(vae, vae.encoder(X)).^2) - 1, 1))

"""
	rerr(vae, X, L)

Reconstruction error.
"""
#rerr(vae::VAE, X) = Flux.mse(vae(X), X)
rerr(vae::VAE, X, L) = Flux.mse(mean([vae(X) for l in 1:L]), X)

"""
	loss(vae, X, L, lambda)

Loss function of the variational autoencoder. Lambda is scaling parameter of 
the KLD, 1 = full KL, 0 = no KL (vanilla autoencoder).
"""
loss(vae::VAE, X, L, lambda) = rerr(vae, X, L) + lambda*KL(vae, X)

"""
	evalloss(vae, X, L, lambda)

Print vae loss function values.	
"""
evalloss(vae::VAE, X, L, lambda) = print("loss: ", loss(vae, X, L, lambda).tracker.data, 
	"\nreconstruction error: ", rerr(vae, X, L).tracker.data, 
	"\nKL: ", KL(vae, X).tracker.data, "\n\n")

"""
	fit!(vae, X, L, M, [iterations, cbit, verb, lambda, rdelta, traindata])

Trains the VAE neural net.
vae - a VAE object
X - data array with instances as columns
L - snumber of samples for KLD computation
M - batchsize
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
lambda - scaling for the KLD loss
rdelta - stopping condition for reconstruction error
traindata - a dictionary for training progress control
"""
function fit!(vae::VAE, X, L, M; iterations=1000, cbit = 200, verb = true, lambda = 1, 
	rdelta = Inf, traindata = nothing)
	# settings
	opt = ADAM(params(vae))

	# train
	for i in 1:iterations
		# sample minibatch of X
		x = X[:, sample(1:size(X,2), M, replace = false)]

		# gradient computation and update
		l = loss(vae, x, L, lambda)
		Flux.Tracker.back!(l)
		opt()

		# callback
		if verb && i%cbit == 0
			evalloss(vae, x, L, lambda)
		end

		# save actual iteration data
		if traindata != nothing
			track!(vae, traindata, x, L, lambda)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = rerr(vae, x, L).tracker.data
			if re < rdelta
				println("Training ended prematurely after $i iterations,\n",
					"reconstruction error $re < $rdelta")
				break
			end
		end
	end
end

"""
	track!(vae, traindata, X, L, lambda)

Save current progress.
"""
function track!(vae::VAE, traindata, X, L, lambda)
	# loss
	if haskey(traindata, "loss")
		push!(traindata["loss"], Flux.Tracker.data(loss(vae, X, L, lambda)))
	else
		traindata["loss"] = [Flux.Tracker.data(loss(vae, X, L, lambda))]
	end

	# KLD
	if haskey(traindata, "KLD")
		push!(traindata["KLD"], Flux.Tracker.data(KL(vae, X)))
	else
		traindata["KLD"] = [Flux.Tracker.data(KL(vae, X))]
	end

	# reconstruction error
	if haskey(traindata, "reconstruction error")
		push!(traindata["reconstruction error"], Flux.Tracker.data(rerr(vae, X, L)))
	else
		traindata["reconstruction error"] = [Flux.Tracker.data(rerr(vae, X, L))]
	end
end


##################
### vae output ###
##################

"""
	getcode(vae, X)

Produces code z for given X.
"""
getcode(vae::VAE, X) = vae.sampler(vae, vae.encoder(X))

"""
	generate(vae)

Generate a sample from the posterior.
"""
generate(vae::VAE) = vae.decoder(randn(Int(size(vae.encoder.layers[end].W,1)/2))).data

"""
	generate(vae, n)

Generate n samples.
"""
generate(vae::VAE, n::Int) = vae.decoder(randn(Int(size(vae.encoder.layers[end].W,1)/2),n)).data

######################
### classification ###
######################

"""
	anomalyscore(vae, X, L)

Compute anomaly score for X.
"""
anomalyscore(vae::VAE, X, L) = rerr(vae, X, L)

"""
	classify(vae, x, threshold, L)

Classify an instance x using reconstruction error and threshold.
"""
classify(vae::VAE, x, threshold, L) = (anomalyscore(vae, x, L) > threshold)? 1 : 0
classify(vae::VAE, x::Array{Float64,1}, threshold, L) = (anomalyscore(vae, x, L) > threshold)? 1 : 0
classify(vae::VAE, X::Array{Float64,2}, threshold, L) = reshape(mapslices(y -> classify(vae, y, threshold, L), X, 1), size(X,2))

"""
	getthreshold(vae, x, L, contamination, [beta])

Compute threshold for VAE classification based on known contamination level.
"""
function getthreshold(vae::VAE, x, L, contamination; Beta = 1.0)
	N = size(x, 2)
	# get reconstruction errors
	xerr  = mapslices(y -> anomalyscore(vae, y, L), x, 1)
	# create ordinary array from the tracked array
	xerr = reshape([e.tracker.data for e in xerr], N)
	# sort it
	xerr = sort(xerr)
	aN = max(Int(floor(N*contamination)),1) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	return Beta*xerr[end-aN] + (1-Beta)*xerr[end-aN+1]
end

##############################################################################
### A SK-learn like model based on VAE with the same methods and some new. ###
##############################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct VAEmodel
	vae::VAE
	lambda::Real
	threshold::Real
	contamination::Real
	iterations::Int
	cbit::Real
	verbfit::Bool
	L::Int # for better precision of classification
	M::Int # batchsize
	rdelta::Float64
	Beta::Float64
	traindata
end

"""
	VAEmodel(esize, dsize, lambda, threshold, contamination, iteration, 
	L, cbit, [activation, rdelta, Beta, tracked])

Initialize a variational autoencoder model with given parameters.
"""
function VAEmodel(esize::Array{Int64,1}, dsize::Array{Int64,1},
	lambda::Real, threshold::Real, contamination::Real, iterations::Int, 
	cbit::Int, verbfit::Bool, L::Int, M::Int; activation = Flux.relu, rdelta = Inf, 
	Beta = 1.0, tracked = false)
	# construct the AE object
	vae = VAE(esize, dsize, activation = activation)
	(tracked)? traindata = Dict{Any, Any}() : traindata = nothing
	model = VAEmodel(vae, lambda, threshold, contamination, iterations, cbit, verbfit, 
		L, M, rdelta, Beta, traindata)
	return model
end

# reimplement some methods of VAE
(model::VAEmodel)(x) = model.vae(x)   
mu(model::VAEmodel, X) = mu(model.vae, model.vae.encoder(X))
sigma(model::VAEmodel, X) = sigma(model.vae, model.vae.encoder(X))
sample_z(model::VAEmodel, X) = sample_z(model.vae, model.vae.encoder(X))
getcode(model::VAEmodel, X) = getcode(model.vae, X)
KL(model::VAEmodel, X) = KL(model.vae, X)
rerr(model::VAEmodel, X) = rerr(model.vae, X, model.L)
loss(model::VAEmodel, X) = loss(model.vae, X, model.L, model.lambda)
evalloss(model::VAEmodel, X) = evalloss(model.vae, X, model.L, model.lambda) 
generate(model::VAEmodel) = generate(model.vae)
generate(model::VAEmodel, n::Int) = generate(model.vae, n)
classify(model::VAEmodel, x) = classify(model.vae, x, model.threshold, model.L)
getthreshold(model::VAEmodel, x) = getthreshold(model.vae, x, model.L, model.contamination, Beta = model.Beta)

"""
	plot(model)

Plot the model losses.
"""
function plot(model::VAEmodel)
	# plot model loss
	if model.traindata == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
	    figure()
	    title("model loss, lambda = $(model.lambda)")
	    y1, = plot(model.traindata["loss"], label = "loss")
	    y2, = plot(model.traindata["reconstruction error"], label = "reconstruction error")
	    ax = gca()
	    ylabel("loss + reconstruction error")
	    xlabel("iteration")
	    
	    ax2 = ax[:twinx]()
	    y3, = plot(model.traindata["KLD"], label = "KLD", c = "g")
	    ylabel("KLD")
	    legend([y1, y2, y3], ["loss", "reconstruction error", "KLD"])
	    show()
	end
end

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
function fit!(model::VAEmodel, X, Y) 
	# train the NN only on normal samples
	nX = X[:, Y.==0]

	# fit the VAE NN
	fit!(model.vae, nX, model.L, model.M, iterations = model.iterations, 
	cbit = model.cbit, verb = model.verbfit, lambda = model.lambda, 
	rdelta = model.rdelta, traindata = model.traindata)

	# now set the threshold using contamination rate
	model.contamination = size(Y[Y.==1],1)/size(Y[Y.==0],1)
	setthreshold!(model, X)
end

"""
	predict(model, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::VAEmodel, X) 
	return classify(model, X)
end
