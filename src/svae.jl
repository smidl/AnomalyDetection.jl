# implementation of symetric variational autoencoder, or sVAE-r, as in https://arxiv.org/abs/1709.01846
#####################
### symmetric VAE ###
#####################

"""
    sVAE

Flux-like structure for the symmetric variational autoencoder.
"""
struct sVAE
    encoder # encoder
    sampler 
    decoder# decoder
    discriminator # discriminator
    _discriminator # discriminator view
end

# make it trainable
Flux.treelike(sVAE)

# make it callable - produce the vae output
(svae::sVAE)(X) = svae.decoder(svae.sampler(svae, svae.encoder(X)))

"""
	mu(svae, X)

Extract means from the last encoder layer.
"""
mu(svae::sVAE, X) = X[1:Int(size(svae.encoder.layers[end].W,1)/2),:]

"""
	sigma(svae, X)

Extract sigmas from the last encoder layer.
"""
sigma(svae::sVAE, X) = softplus(X[Int(size(svae.encoder.layers[end].W,1)/2+1):end,:]) + 1e-6

"""
	sample_z(svae, X)

Sample from the last encoder layer.
"""
function sample_z(svae::sVAE, X) 
    res = mu(svae,X)
    return res + randn(size(res)) .* sigma(svae,X) 
end

"""
	sVAE(ensize, decsize, dissize; [activation])

Initialize a variational autoencoder with given encoder size and decoder size.
ensize - vector of ints specifying the width anf number of layers of the encoder
decsize - size of decoder
dissize - size of discriminator
activation - arbitrary activation function
"""
function sVAE(ensize::Array{Int64,1}, decsize::Array{Int64,1}, dissize::Array{Int64,1};
        activation = Flux.relu)
    @assert size(ensize, 1) >= 3
    @assert size(decsize, 1) >= 3
    @assert size(dissize, 1) >= 3
    @assert ensize[end] == 2*decsize[1] 
    @assert ensize[1] == decsize[end]
    @assert ensize[1] + decsize[1] == dissize[1]
    @assert dissize[end] == 1

    # construct the encoder
    encoder = Dense(ensize[1],ensize[2],activation)
    for i in 3:(size(ensize,1)-1)
        encoder = Chain(encoder, Dense(ensize[i-1],ensize[i],activation))
    end
    encoder = Chain(encoder, Dense(ensize[end-1], ensize[end]))

    # construct the decoder
    decoder = Dense(decsize[1],decsize[2],activation)
    for i in 3:(size(decsize,1)-1)
        decoder = Chain(decoder, Dense(decsize[i-1],decsize[i],activation))
    end
    decoder = Chain(decoder, Dense(decsize[end-1], decsize[end]))

    # construct the discriminator
    discriminator = Dense(dissize[1],dissize[2],activation)
    for i in 3:(size(dissize,1)-1)
        discriminator = Chain(discriminator, Dense(dissize[i-1],dissize[i],activation))
    end
    discriminator = Chain(discriminator, Dense(dissize[end-1], dissize[end]))

    # finally construct the svae struct
    svae = sVAE(encoder, sample_z, decoder, discriminator, freeze(discriminator))

    return svae
end

"""
    distrain(svae, X, Z)

Produces the discriminator output from X and Z, for discriminator training purposes.
"""
function distrain(svae::sVAE, X, Z)
    xdim = size(X,1)
    zdim = size(Z,1)
    xzcat = cat(1, eye(xdim), zeros(zdim, xdim))*X + 
        cat(1, zeros(xdim, zdim), eye(zdim))*Z
    return svae.discriminator(xzcat) 
end

"""
    disfix(svae, X, Z)

Produces the discriminator output from X and Z, non-trainable version.
"""
function disfix(svae::sVAE, X, Z)
    xdim = size(X,1)
    zdim = size(Z,1)
    xzcat = cat(1, eye(xdim), zeros(zdim, xdim))*X + 
        cat(1, zeros(xdim, zdim), eye(zdim))*Z
    return svae._discriminator(xzcat) 
end

"""
    discriminate(svae, X, Z)

Discriminator output after sigmoid transform, trainable!
"""
discriminate(svae::sVAE, X, Z) = σ.(distrain(svae, X, Z))

################
### training ###
################

"""
    Dloss(svae, pX, pZ, qX, qZ)

sVAE discriminator loss.
"""
Dloss(svae::sVAE, pX, pZ, qX, qZ) = -mean(log.(1-σ.(distrain(svae,qX, qZ)))) - 
    mean(log.(σ.(distrain(svae, pX, pZ))))

"""
    Dloss(svae, X, L)

sVAE discriminator loss, L is batchsize.
"""
function Dloss(svae::sVAE, X, L::Int64)
    N = size(X, 2)
    latentdim = Int(size(svae.encoder.layers[end].W,1)/2)

    # sample q(x, z)
    qX = X[:, sample(1:N, L, replace = false)]
    qZ = Flux.Tracker.data(getcode(svae, qX))

    # sample p(x,z)
    pZ = randn(latentdim, N)
    pX = Flux.Tracker.data(svae.decoder(pZ))
    
    return Dloss(svae, pX, pZ, qX, qZ)
end

"""
    VAEloss(svae, X, L, lambda; xsigma = 1.0)

Variational loss for sVAE.
"""
function VAEloss(svae::sVAE, X, L, lambda; xsigma = 1.0)
    N = size(X, 2)
    latentdim = Int(size(svae.encoder.layers[end].W,1)/2)
    
    # sample q(x,z)
    qX = X[:, sample(1:N, L, replace = false)]
    qZ = getcode(svae, qX)

    # sample p(x,z)
    pZ = randn(latentdim, L)
    pX = svae.decoder(pZ)
    
    # also, gather the params of q(z|x)
    zmu = mu(svae, svae.encoder(pX))
    zsigma = sigma(svae, svae.encoder(pX))
    
    # is this entirely correct?
    return -(mean(disfix(svae, qX, qZ)) - 
        #1/xsigma*lambda*Flux.mse(qX, pX) - # this converges the best
        1/xsigma*lambda*Flux.mse(qX, svae.decoder(Flux.Tracker.data(qZ))) - # but this is probably correct
        mean(disfix(svae, pX, pZ)) - mean((zmu - pZ).^2./zsigma))
end

"""
    rerr(svae, X, M)

sVAE reconstruction error, M is number of samples.
"""
rerr(svae::sVAE, X, M) = Flux.mse(mean([svae(X) for m in 1:M]), X)

"""
    evalloss(svae, X, L, lambda, M)
"""
evalloss(svae::sVAE, X, L, lambda, M) = println(
    "discriminator loss: ", Flux.Tracker.data(Dloss(svae, X, L)),
    "\nVAE loss: ", Flux.Tracker.data(VAEloss(svae, X, L, lambda)),
    "\nreconstruction error: ", Flux.Tracker.data(rerr(svae, X, M)), "\n"
    )

"""
    fit!(svae, X, L, lambda, [M, iterations, cbit, verb, rdelta, traindata])

Trains the sVAE neural net.
svae - a sVAE object
X - data array with instances as columns
L - batchsize
lambda - scaling for the p(x|z) and q(z|x) logs, >= 0
M - sampling repetition
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
traindata - a dictionary for training progress control
"""
function fit!(svae::sVAE, X, L, lambda; M=1, iterations=1000, cbit = 200, 
        verb = true, rdelta = Inf, traindata = nothing)
    # settings
    opt = ADAM(params(svae))

    # train
    for i in 1:iterations
        # train the discriminator
        Dl = Dloss(svae, X, L)
        Flux.Tracker.back!(Dl)
        opt()
        
        # train the VAE part of the net
        Vl = VAEloss(svae, X, L, lambda)
        Flux.Tracker.back!(Vl)
        opt()

        # callback
        if i%cbit == 0
            evalloss(svae, X, L, lambda, M)
        end

        # save actual iteration data
        if traindata != nothing
            track!(svae, traindata, X, L, lambda, M)
        end

        # if stopping condition is present
        if rdelta < Inf
            re = Flux.Tracker.data(rerr(svae, X, M))
            if re < rdelta
                println("Training ended prematurely after $i iterations,\n",
                    "reconstruction error $re < $rdelta")
                break
            end
        end
    end
end

"""
    track!(svae, traindata, X, L, lambda, M)

Save current progress.
"""
function track!(svae::sVAE, traindata, X, L, lambda, M)
    # discriminator loss
    if haskey(traindata, "discriminator loss")
        push!(traindata["discriminator loss"], Flux.Tracker.data(Dloss(svae, X, L)))
    else
        traindata["discriminator loss"] = [Flux.Tracker.data(Dloss(svae, X, L))]
    end

    # VAE loss
    if haskey(traindata, "VAE loss")
        push!(traindata["VAE loss"], Flux.Tracker.data(VAEloss(svae, X, L, lambda)))
    else
        traindata["VAE loss"] = [Flux.Tracker.data(VAEloss(svae, X, L, lambda))]
    end

    # reconstruction error
    if haskey(traindata, "reconstruction error")
        push!(traindata["reconstruction error"], Flux.Tracker.data(rerr(svae, X, M)))
    else
        traindata["reconstruction error"] = [Flux.Tracker.data(rerr(svae, X, M))]
    end
end

###################
### svae output ###
###################

"""
    getcode(svae, X)

Produces code z for given X.
"""
getcode(svae::sVAE, X) = svae.sampler(svae, svae.encoder(X))

"""
    generate(svae)

Generate a sample from the posterior.
"""
generate(svae::sVAE) = svae.decoder(randn(Int(size(svae.encoder.layers[end].W,1)/2))).data

"""
    generate(svae, n)

Generate n samples.
"""
generate(svae::sVAE, n::Int) = svae.decoder(randn(Int(size(svae.encoder.layers[end].W,1)/2),n)).data

######################
### classification ###
######################

"""
    anomalyscore(svae, X, M, alpha)

Compute anomaly score for X, M is sampling repetition.
alpha - weighs between reconstruction error and discriminator term
"""
anomalyscore(svae::sVAE, X, M, alpha) = alpha*rerr(svae, X, M) + (1-alpha)*mean(discriminate(svae, X, getcode(svae, X)))

"""
    classify(svae, x, threshold, M, alpha)

Classify an instance x using reconstruction error and threshold.
"""
classify(svae::sVAE, x, threshold, M, alpha) = (anomalyscore(svae, x, M, alpha) > threshold)? 1 : 0
classify(svae::sVAE, x::Array{Float64,1}, threshold, M, alpha) = (anomalyscore(svae, x, M, alpha) > threshold)? 1 : 0
classify(svae::sVAE, X::Array{Float64,2}, threshold, M, alpha) = reshape(mapslices(y -> classify(svae, y, threshold, M, alpha), X, 1), size(X,2))

"""
    getthreshold(svae, x, M, alpha, contamination, [beta])

Compute threshold for sVAE classification based on known contamination level.
"""
function getthreshold(svae::sVAE, x, M, alpha, contamination; Beta = 1.0)
    N = size(x, 2)
    # get reconstruction errors
    xerr  = mapslices(y -> anomalyscore(svae, y, M, alpha), x, 1)
    # create ordinary array from the tracked array
    xerr = reshape([Flux.Tracker.data(e)[1] for e in xerr], N)
    # sort it
    xerr = sort(xerr)
    aN = max(Int(floor(N*contamination)),1) # number of contaminated samples
    # get the threshold - could this be done more efficiently?
    return Beta*xerr[end-aN] + (1-Beta)*xerr[end-aN+1]
end

###############################################################################
### A SK-learn like model based on sVAE with the same methods and some new. ###
###############################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct sVAEmodel
    svae::sVAE
    lambda::Real
    threshold::Real
    contamination::Real
    iterations::Int
    cbit::Real
    L::Int # batchsize
    M::Int # sampling rate for reconstruction error
    verbfit::Bool
    rdelta::Float64
    alpha::Float64
    Beta::Float64
    xsigma::Float64
    traindata
end

"""
    sVAEmodel(ensize, decsize, dissize, lambda, threshold, contamination, iterations, 
    cbit, verbfit, L, [M, activation, rdelta, alpha, Beta, xsigma, tracked])

Initialize a sVAE model with given parameters.

ensize - encoder architecture
decsize - decoder architecture
dissize - discriminator architecture
lambda - weight of the data reconstruction term in the total loss
threshold - anomaly score threshold for classification, is set automatically using contamination during fit
contamination - percentage of anomalous samples in all data for automatic threshold computation
iterations - number of training iterations
cbit - current training progress is printed every cbit iterations
verbfit - is progress printed?
L - batchsize
M [1] - number of samples taken during computation of reconstruction error, higher may produce more stable classification results
activation [Flux.relu] - activation function
rdelta [Inf] - training stops if reconstruction error is smaller than rdelta
alpha [0.5] - weighs between the reconstruction error (1) and discriminator score (0) in classification
Beta [1.0] - how tight around normal data is the automatically computed threshold
xsigma [1.0] - static estiamte of data variance
tracked [false] - is training progress (losses) stored?
"""
function sVAEmodel(ensize::Array{Int64,1}, decsize::Array{Int64,1},
    dissize::Array{Int64,1}, lambda::Real, threshold::Real, contamination::Real, 
    iterations::Int, cbit::Int, verbfit::Bool, L::Int; M=1, activation = Flux.relu, 
    rdelta = Inf, alpha=0.5, Beta = 1.0, xsigma = 1.0, tracked = false)
    # construct the AE object
    svae = sVAE(ensize, decsize, dissize, activation = activation)
    (tracked)? traindata = Dict{Any, Any}() : traindata = nothing
    model = sVAEmodel(svae, lambda, threshold, contamination, iterations, cbit, L, M, 
        verbfit, rdelta, alpha, Beta, xsigma, traindata)
    return model
end

# reimplement some methods of sVAE
(model::sVAEmodel)(x) = model.svae(x)   
mu(model::sVAEmodel, X) = mu(model.svae, model.svae.encoder(X))
sigma(model::sVAEmodel, X) = sigma(model.svae, model.svae.encoder(X))
sample_z(model::sVAEmodel, X) = sample_z(model.svae, model.svae.encoder(X))
getcode(model::sVAEmodel, X) = getcode(model.svae, X)
discriminate(model::sVAEmodel, X, Z) = discriminate(model.svae, X, Z)
Dloss(model::sVAEmodel, pX, pZ, qX, qZ) = Dloss(model.svae, pX, pZ, qX, qZ)
Dloss(model::sVAEmodel, X) = Dloss(model.svae, X, model.L)
VAEloss(model::sVAEmodel, X) = VAEloss(model.svae, X, model.L, model.lambda, xsigma = model.xsigma)
rerr(model::sVAEmodel, X) = rerr(model.svae, X, model.M)
evalloss(model::sVAEmodel, X) = evalloss(model.svae, X, model.L, model.lambda, model.M)
generate(model::sVAEmodel) = generate(model.svae)
generate(model::sVAEmodel, n) = generate(model.svae, n)
anomalyscore(model::sVAEmodel, X) = anomalyscore(model.svae, X, model.M, model.alpha)
classify(model::sVAEmodel, X) = classify(model.svae, X, model.threshold, model.M, model.alpha)
getthreshold(model::sVAEmodel, x) = getthreshold(model.svae, x, model.M, model.alpha, model.contamination, Beta = model.Beta)

"""
    plot(model)

Plot the model losses.
"""
function plot(model::sVAEmodel)
    # plot model loss
    if model.traindata == nothing
        println("No data to plot, set traindata = Dict{Any, Any}() to record training.")
        return
    else
        figure()
        title("model loss, lambda = $(model.lambda)")
        y1, = plot(model.traindata["discriminator loss"], label = "discriminator loss")
        y2, = plot(model.traindata["VAE loss"], label = "VAE loss", c = "g")
        ax = gca()
        ylabel("discriminator loss + VAE loss")
        xlabel("iteration")

        ax2 = ax[:twinx]()
        y3, = plot(model.traindata["reconstruction error"], c = "r", label = "reconstruction error")
        ylabel("reconstruction error")
        legend([y1, y3, y2], ["discriminator loss", "reconstruction error", "VAE loss"])
        show()
    end
end

"""
    setthreshold!(model, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::sVAEmodel, X)
    model.threshold = getthreshold(model, X)
end

"""
    fit!(model, X, Y)

Trains a sVAEmodel.
"""
function fit!(model::sVAEmodel, X, Y)
    # train the NN only on normal samples
    nX = X[:, Y.==0]

    # train the sVAE NN
    fit!(model.svae, nX, model.L, model.lambda, M=model.M,
     iterations=model.iterations, cbit = model.cbit, verb = model.verbfit, 
     rdelta = model.rdelta, traindata = model.traindata)
    
    # now set the threshold using contamination rate
    model.contamination = size(Y[Y.==1],1)/size(Y[Y.==0],1)
    setthreshold!(model, X)
end

"""
    predict(model, X)

Based on known contamination level, compute threshold and classify instances in X.
"""
function predict(model::sVAEmodel, X) 
    return classify(model, X)
end
