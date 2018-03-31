##############
### basics ###
##############

"""
	GAN

Flux-like object representing a generative adversarial network.
"""
struct GAN
	g # generator
	gg # non-trainable generator copy
	d # discriminator
	dd # non-trainable discriminator copy
	pz # code distribution
end


"""
	GAN(generator, discriminator)

Basic GAN constructor.
"""
GAN(G::Flux.Chain, D::Flux.Chain; pz=randn) = GAN(G, freeze(G), D, freeze(D), pz)

"""
	GAN(gsize, dsize, [pz, activation])

Constructor for the GAN object. 

gsize - vector of Ints describing generator layers sizes
dsize - vector of Ints describing discriminator layers sizes, including the last scalar layer 
pz - code distribution
activation - activation function common to all layers
"""
function GAN(gsize, dsize; pz = randn, activation = Flux.leakyrelu)
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

	return GAN(generator, discriminator, pz=pz)
end

################
### training ###
################

"""
	Dloss(gan, X, Z)

Discriminator loss.
"""
Dloss(gan::GAN, X, Z) = - 0.5*(mean(log.(gan.d(X))) + mean(log.(1 - gan.d(gan.gg(Z)))))

"""
	Gloss(gan, Z)

Generator loss.
"""
Gloss(gan::GAN, Z) = - mean(log.(gan.dd(gan.g(Z))))

"""
	rerr(gan, X, Z)

Crude estimate of reconstruction error.
"""
#rerr(gan::GAN, X, Z) = Flux.mse(mean(gan.g(Z).data,2), mean(X,2))
rerr(gan::GAN, X, Z) = Flux.mse(gan.g(Z), X) # which of these is better?
# the first one can easily get fooled in multimodal setting

"""
	evalloss(gan, X, Z)
"""
evalloss(gan::GAN, X, Z) = print("discriminator loss: ", Dloss(gan, X, Z).tracker.data,  
	"\ngenerator loss: ", Gloss(gan, Z).tracker.data, 
	"\nreconstruction error: ", rerr(gan, X, Z).tracker.data, "\n\n")

"""
	fit!(gan, X, L, [iterations, cbit, verb, rdelta])

Trains a GAN.

gan - struct of type GAN
X - data array with instances as columns
L - batchsize
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
traindata - a dictionary for training progress control
"""
function fit!(gan::GAN, X, L; iterations=1000, cbit = 200, verb = true, rdelta = Inf,
	traindata = nothing)
	# settings
	#Dopt = ADAM(params(gan.d))
	Dopt = SGD(params(gan.d))
	Gopt = ADAM(params(gan.g))
	
	# problem size
	N = size(X,2)
	zdim = size(params(gan.g)[1],2)

	# train the GAN
	for i in 1:iterations
		# sample data and generate codes
		x = X[:,sample(1:N, L, replace=false)]
		z = getcode(gan, L)
                
        # discriminator training
        Dl = Dloss(gan, x,z)
        Flux.Tracker.back!(Dl)
        Dopt()
		
		# generator training	
        Gl = Gloss(gan, z)
        Flux.Tracker.back!(Gl)
        Gopt()
	
		# callback
		if verb && i%cbit==0
			evalloss(gan, x, z)
		end

		# save actual iteration data
		if traindata != nothing
			track!(gan, traindata, x, z)
		end

		# if a stopping condition is present
		if rdelta < Inf
			re = rerr(gan, x, z) 
			if re < rdelta
				println("Training ended prematurely after $i iterations,
					\nreconstruction error $re < $rdelta")
				break
			end
		end
	end
end

"""
	track!(gan, traindata, X, Z)

Save current progress.
"""
function track!(gan::GAN, traindata, X, Z)
	# Dloss
	if haskey(traindata, "discriminator loss")
		push!(traindata["discriminator loss"], Flux.Tracker.data(Dloss(gan, X, Z)))
	else
		traindata["discriminator loss"] = [Flux.Tracker.data(Dloss(gan, X, Z))]
	end

	# Gloss
	if haskey(traindata, "generator loss")
		push!(traindata["generator loss"], Flux.Tracker.data(Gloss(gan, Z)))
	else
		traindata["generator loss"] = [Flux.Tracker.data(Gloss(gan, Z))]
	end

	# reconstruction error
	if haskey(traindata, "reconstruction error")
		push!(traindata["reconstruction error"], Flux.Tracker.data(rerr(gan, X, Z)))
	else
		traindata["reconstruction error"] = [Flux.Tracker.data(rerr(gan, X, Z))]
	end
end

############################
### auxilliary functions ###
############################
 
"""
	getcode(gan)

Generate a sample code from GAN.
"""
getcode(gan::GAN) = gan.pz(size(params(gan.g)[1],2))

"""
	getcode(gan, N)

Generate codes from GAN.
"""
getcode(gan::GAN, n::Int) = gan.pz(size(params(gan.g)[1],2), n)

"""
	generate(gan)

Generate one sample.
"""
generate(gan::GAN) = gan.g(getcode(gan)).data

"""
	generate(gan, n)

Generate n samples.
"""
generate(gan::GAN, n::Int) = gan.g(getcode(gan, n)).data

"""
    discriminate(gan, X)

Return discrimiantor score.
"""
discriminate(gan::GAN, X) = Flux.Tracker.data(gan.d(X))

######################
### classification ###
######################

"""
	anomalyscore(gan, X, lambda)

Computes the anomaly score of X under given GAN.
"""
#anomalyscore(gan::GAN, X, lambda) = (1 - lambda)*-mean(log.(gan.d(X))) + lambda*Flux.mse(mean(generate(gan, size(X,2)),2), mean(X,2))
#anomalyscore(gan::GAN, X, lambda) = (1 - lambda)*-mean(log.(gan.d(X))) + lambda*Flux.mse(generate(gan, size(X,2)), X)
anomalyscore(gan::GAN, X, lambda) = (1 - lambda)*-mean(log.(gan.d(X))).tracker.data + lambda*rerr(gan, X, getcode(gan, size(X,2)))

"""
	classify(gan, x, threshold, lambda)

Classify an instance x using the discriminator and a threshold.
"""
classify(gan::GAN, x, threshold, lambda) = (anomalyscore(gan, x, lambda) > threshold)? 1 : 0
classify(gan::GAN, x::Array{Float64,1}, threshold, lambda) = (anomalyscore(gan, x, lambda) > threshold)? 1 : 0
classify(gan::GAN, X::Array{Float64,2}, threshold, lambda) = reshape(mapslices(y -> classify(gan, y, threshold, lambda), X, 1), size(X,2))

"""
	getthreshold(gan, x, contamination, lambda, [Beta])

Compute threshold for GAN classification based on known contamination level.
"""
function getthreshold(gan::GAN, X, contamination, lambda; Beta = 1.0)
	N = size(X, 2)
	# get anomaly score
	ascore = mapslices(y -> anomalyscore(gan, y, lambda), X, 1)
	# create ordinary array from the tracked array
	ascore = reshape([s for s in ascore], N)
	# sort it
	ascore = sort(ascore)
	aN = max(Int(floor(N*contamination)),1) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	return Beta*ascore[end-aN] + (1-Beta)ascore[end-aN+1]
end

##############################################################################
### A SK-learn like model based on GAN with the same methods and some new. ###
##############################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct GANmodel
	gan::GAN
	lambda::Real
	threshold::Real
	contamination::Real
	L::Int
	iterations::Int
	cbit::Real
	verbfit::Bool
	rdelta::Float64
	Beta::Float64
	traindata
end

"""
	GANmodel(gsize, dsize, lambda, threshold, contamination, L, iterations, 
	cbit, verbfit, [pz, activation, rdelta, Beta, tracked])

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
Beta [1.0] - how tight around normal data is the automatically computed threshold
tracked [false] - is training progress (losses) stored?
"""
function GANmodel(gsize::Array{Int64,1}, dsize::Array{Int64,1},
	lambda::Real, threshold::Real, contamination::Real, L::Int, iterations::Int, 
	cbit::Int, verbfit::Bool; pz = randn, activation = Flux.leakyrelu, rdelta = Inf,
	Beta = 1.0, tracked = false)
	# construct the AE object
	gan = GAN(gsize, dsize, pz = pz, activation = activation)
	(tracked)? traindata = Dict{Any, Any}() : traindata = nothing
	model = GANmodel(gan, lambda, threshold, contamination, L, iterations, cbit, 
		verbfit, rdelta, Beta, traindata)
	return model
end

# reimplement some methods of GAN
Dloss(model::GANmodel, X, Z) = Dloss(model.gan, X, Z)
Gloss(model::GANmodel, Z) = Gloss(model.gan, Z)
rerr(model::GANmodel, X, Z) = rerr(model.gan, X, Z)
evalloss(model::GANmodel, X, Z) = evalloss(model.gan, X, Z) 
generate(model::GANmodel) = generate(model.gan)
generate(model::GANmodel, n::Int) = generate(model.gan, n)
anomalyscore(model::GANmodel, X) = anomalyscore(model.gan, X, model.lambda)
classify(model::GANmodel, x) = classify(model.gan, x, model.threshold, model.lambda)
classify(model::GANmodel, x::Array{Float64,1}) = classify(model.gan, x, model.threshold, model.lambda)
classify(model::GANmodel, X::Array{Float64,2}) = classify(model.gan, X, model.threshold, model.lambda)
getthreshold(model::GANmodel, X) = getthreshold(model.gan, X, model.contamination, model.lambda, Beta = model.Beta)
getcode(model::GANmodel) = getcode(model.gan)
getcode(model::GANmodel, n) = getcode(model.gan, n)
discriminate(model::GANmodel, X) = discriminate(model.gan, X)

"""
	plot(model)

Plot the model losses.
"""
function plot(model::GANmodel)
	# plot model loss
	if model.traindata == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
	    figure()
	    title("model loss")
	    y1, = plot(model.traindata["generator loss"], label = "generator loss")
	    y2, = plot(model.traindata["discriminator loss"], label = "discriminator loss")
	    ylabel("Gloss + Dloss")
	    xlabel("iteration")
	    ax = gca()
	    
	    ax2 = ax[:twinx]()
	    y3, = plot(model.traindata["reconstruction error"], label = "reconstruction error", c = "g")
	    ylabel("reconstruction error")
	    legend([y1, y2, y3], ["generator loss", "discriminator loss", "reconstruction error"])
	    show()
	end
end

"""
	setthreshold!(model, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::GANmodel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model, X, Y)

Trains a GANmodel.
"""
function fit!(model::GANmodel, X, Y)
	# train the NN only on normal samples
	nX = X[:, Y.==0]

	# train the GAN NN
	fit!(model.gan, nX, model.L; iterations=model.iterations, 
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
function predict(model::GANmodel, X) 
	return classify(model, X)
end
