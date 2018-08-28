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
sigma(svae::sVAE, X) = softplus(X[Int(size(svae.encoder.layers[end].W,1)/2+1):end,:]) + Float(1e-6)

"""
	sample_z(svae, X)

Sample from the last encoder layer.
"""
function sample_z(svae::sVAE, X) 
    μ = mu(svae,X)
    σ = sigma(svae, X)
    ϵ = Float.(randn(size(μ)))
    return μ + ϵ.*σ 
end

"""
	sVAE(ensize, decsize, dissize; [activation, layer])

Initialize a variational autoencoder with given encoder size and decoder size.
ensize - vector of ints specifying the width anf number of layers of the encoder
decsize - size of decoder
dissize - size of discriminator
activation - arbitrary activation function
layer - type of layer
"""
function sVAE(ensize::Array{Int64,1}, decsize::Array{Int64,1}, dissize::Array{Int64,1};
        activation = Flux.relu, layer = Flux.Dense)
    @assert size(ensize, 1) >= 3
    @assert size(decsize, 1) >= 3
    @assert size(dissize, 1) >= 3
    @assert ensize[end] == 2*decsize[1] 
    @assert ensize[1] == decsize[end]
    @assert ensize[1] + decsize[1] == dissize[1]
    @assert dissize[end] == 1

    # construct the encoder
    encoder = aelayerbuilder(ensize, activation, layer)

    # construct the decoder
    decoder = aelayerbuilder(decsize, activation, layer)

    # construct the discriminator
    discriminator = aelayerbuilder(dissize, activation, layer)

    # finally construct the svae struct
    svae = sVAE(encoder, sample_z, decoder, discriminator, freeze(discriminator))

    return svae
end

"""
    xzcat(X, Z)

Concatenates X and Z in the direction of first dimension using matrix multiplication.
"""
function xzcat(X, Z)
    xdim = size(X,1)
    zdim = size(Z,1)
    return cat(1, (eye(Float, xdim)), zeros(Float, zdim, xdim))*X + 
        cat(1, zeros(Float, xdim, zdim), eye(Float, zdim))*Z
end

"""
    distrain(svae, X, Z)

Produces the discriminator output from X and Z, for discriminator training purposes.
"""
distrain(svae::sVAE, X, Z) = svae.discriminator(xzcat(X,Z)) 

"""
    disfix(svae, X, Z)

Produces the discriminator output from X and Z, non-trainable version.
"""
disfix(svae::sVAE, X, Z) = svae._discriminator(xzcat(X,Z)) 

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
Dloss(svae::sVAE, pX, pZ, qX, qZ) = -mean(log.(Float(1)-σ.(distrain(svae,qX, qZ)) + eps(Float))) - 
    mean(log.(σ.(distrain(svae, pX, pZ)) + eps(Float)))

"""
    Dloss(svae, X)

sVAE discriminator loss.
"""
function Dloss(svae::sVAE, X)
    N = size(X, 2)
    latentdim = Int(size(svae.encoder.layers[end].W,1)/2)

    # sample q(x, z)
    qX = X
    qZ = Flux.Tracker.data(getcode(svae, qX))

    # sample p(x,z)
    pZ = randn(Float, latentdim, N)
    pX = Flux.Tracker.data(svae.decoder(pZ))
    
    return Dloss(svae, pX, pZ, qX, qZ)
end

"""
    VAEloss(svae, X, lambda; xsigma = 1.0)

Variational loss for sVAE.
"""
function VAEloss(svae::sVAE, X, lambda; xsigma = 1.0)
    N = size(X, 2)
    latentdim = Int(size(svae.encoder.layers[end].W,1)/2)
    xsigma = Float(xsigma)
    lambda = Float(lambda)

    # sample q(x,z)
    qX = X
    qZ = getcode(svae, qX)

    # sample p(x,z)
    pZ = randn(Float, latentdim, N)
    pX = svae.decoder(pZ)
    
    # also, gather the params of q(z|x)
    zmu = mu(svae, svae.encoder(pX))
    zsigma = sigma(svae, svae.encoder(pX))
    
    # is this entirely correct?
    return -(mean(disfix(svae, qX, qZ)) - 
        #1/xsigma*lambda*Flux.mse(qX, pX) - # this converges the best
        1/xsigma*lambda*Flux.mse(qX, svae.decoder(Flux.Tracker.data(qZ))) - # but this is probably correct
        mean(disfix(svae, pX, pZ)) - lambda*mean((zmu - pZ).^2./zsigma))
end

"""
    rerr(svae, X, M)

sVAE reconstruction error, M is number of samples.
"""
rerr(svae::sVAE, X, M) = Flux.mse(mean([svae(X) for m in 1:M]), X)

"""
    evalloss(svae, X, lambda, M)
"""
function evalloss(svae::sVAE, X, lambda, M) 
    dl, vl, r = getlosses(svae, X, lambda, M)
    println(
    "discriminator loss: ", dl,
    "\nVAE loss: ", vl,
    "\nreconstruction error: ", r, "\n"
    )
end

"""
    getlosses(svae, X, M, lambda)

Return the numeric values of current losses.
"""
getlosses(svae::sVAE, X, lambda, M) = (
    Flux.Tracker.data(Dloss(svae, X)),
    Flux.Tracker.data(VAEloss(svae, X, lambda)),
    Flux.Tracker.data(rerr(svae, X, M))
    )

"""
    fit!(svae, X, batchsize, lambda, [M, iterations, cbit, verb, rdelta, history, eta])

Trains the sVAE neural net.
svae - a sVAE object
X - data array with instances as columns
batchsize - batchsize
lambda - scaling for the p(x|z) and q(z|x) logs, >= 0
M - sampling repetition
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
history - a dictionary for training progress control
eta - learning rate
"""
function fit!(svae::sVAE, X, batchsize, lambda; M=1, iterations=1000, cbit = 200, 
        verb = true, rdelta = Inf, history = nothing, eta = 0.001)
    # settings
    opt = ADAM(params(svae), eta)

    # sampler
    sampler = UniformSampler(X,iterations,batchsize)
    # it might be smaller than the original one if there is not enough data
    batchsize = sampler.batchsize 

    # using ProgressMeter 
    if verb
        p = Progress(iterations, 0.3)
        x = next!(sampler)
        reset!(sampler)
        _dl, _vl, _r = getlosses(svae, x, lambda, M)
    end

    # train
    for (i,x) in enumerate(sampler)
        # train the discriminator
        Dl = Dloss(svae, x)
        if isnan(Dl)
            warn("Discriminator loss is NaN, ending fit.")
            return
        end
        Flux.Tracker.back!(Dl)
        opt()
        
        # train the VAE part of the net
        Vl = VAEloss(svae, x, lambda)
        Flux.Tracker.back!(Vl)
        opt()

        # progress
        if verb 
            if (i%cbit == 0 || i == 1)
                _dl, _vl, _r = getlosses(svae, x, lambda, M)
            end
            ProgressMeter.next!(p; showvalues = [(:"discriminator loss",_dl),
                (:"vae loss", _vl),(:"reconstruction error", _r)])
        end

        # save actual iteration data
        if history != nothing
            track!(svae, history, x, lambda, M)
        end

        # if stopping condition is present
        if rdelta < Inf
            re = Flux.Tracker.data(rerr(svae, x, M))[1]
            if re < rdelta
                println("Training ended prematurely after $i iterations,\n",
                    "reconstruction error $re < $rdelta")
                break
            end
        end
    end
end

"""
    track!(svae, history, X, lambda, M)

Save current progress.
"""
function track!(svae::sVAE, history, X, lambda, M)
    push!(history, :discriminator_loss, Flux.Tracker.data(Dloss(svae, X)))
    push!(history, :vae_loss, Flux.Tracker.data(VAEloss(svae, X, lambda)))
    push!(history, :reconstruction_error, Flux.Tracker.data(rerr(svae, X, M)))
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
generate(svae::sVAE) = svae.decoder(Float.(randn(Int(size(svae.encoder.layers[end].W,1)/2)))).data

"""
    generate(svae, n)

Generate n samples.
"""
generate(svae::sVAE, n::Int) = svae.decoder(Float.(randn(Int(size(svae.encoder.layers[end].W,1)/2),n))).data

######################
### classification ###
######################

"""
    anomalyscore(svae, X, M, alpha)

Compute anomaly score for X, M is sampling repetition.
alpha - weighs between reconstruction error and discriminator term
"""
anomalyscore(svae::sVAE, X::Array{Float, 1}, M, alpha) = 
    Flux.Tracker.data(Float(alpha)*rerr(svae, X, M) + Float(1-alpha)*mean(discriminate(svae, X, getcode(svae, X))))
anomalyscore(svae::sVAE, X::Array{Float, 2}, M, alpha) = 
    reshape(mapslices(y -> anomalyscore(svae, y, M, alpha), X, 1), size(X,2))

"""
    classify(svae, x, threshold, M, alpha)

Classify an instance x using reconstruction error and threshold.
"""
classify(svae::sVAE, X, threshold, M, alpha) = Int.(anomalyscore(svae, X, M, alpha) .> Float(threshold))


###############################################################################
### A SK-learn like model based on sVAE with the same methods and some new. ###
###############################################################################

""" 
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct sVAEmodel <: genmodel
    svae::sVAE
    lambda::Real
    threshold::Real
    contamination::Real
    iterations::Int
    cbit::Real
    batchsize::Int # batchsize
    M::Int # sampling rate for reconstruction error
    verbfit::Bool
    rdelta
    alpha
    Beta
    xsigma
    history
    eta
end

"""
    sVAEmodel(ensize, decsize, dissize, [lambda, threshold, contamination, iterations, 
    cbit, verbfit, batchsize, M, activation, rdelta, alpha, Beta, xsigma, tracked, eta])

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
batchsize - batchsize
M [1] - number of samples taken during computation of reconstruction error, higher may produce more stable classification results
activation [Flux.relu] - activation function
layer - NN layer type
rdelta [Inf] - training stops if reconstruction error is smaller than rdelta
alpha [0.5] - weighs between the reconstruction error (1) and discriminator score (0) in classification
Beta [1.0] - how tight around normal data is the automatically computed threshold
xsigma [1.0] - static estiamte of data variance
tracked [false] - is training progress (losses) stored?
eta [0.001] - learning rate
"""
function sVAEmodel(ensize::Array{Int64,1}, decsize::Array{Int64,1},
    dissize::Array{Int64,1}; lambda::Real=0.5, threshold::Real=0.0, 
    contamination::Real=0.0, iterations::Int=10000, cbit::Int=1000, 
    verbfit::Bool=true, batchsize::Int=1, M=1, activation = Flux.relu, 
    layer = Flux.Dense, rdelta = Inf, alpha=0.5, Beta = 1.0, xsigma = 1.0, 
    tracked = false, eta = 0.001)
    # construct the AE object
    svae = sVAE(ensize, decsize, dissize, activation = activation, layer = layer)
    (tracked)? history = MVHistory() : history = nothing
    model = sVAEmodel(svae, lambda, threshold, contamination, iterations, cbit, batchsize, M, 
        verbfit, rdelta, alpha, Beta, xsigma, history, eta)
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
Dloss(model::sVAEmodel, X) = Dloss(model.svae, X)
VAEloss(model::sVAEmodel, X) = VAEloss(model.svae, X, model.lambda, xsigma = model.xsigma)
rerr(model::sVAEmodel, X) = rerr(model.svae, X, model.M)
evalloss(model::sVAEmodel, X) = evalloss(model.svae, X, model.lambda, model.M)
generate(model::sVAEmodel) = generate(model.svae)
generate(model::sVAEmodel, n) = generate(model.svae, n)
anomalyscore(model::sVAEmodel, X) = anomalyscore(model.svae, X, model.M, model.alpha)
classify(model::sVAEmodel, X) = classify(model.svae, X, model.threshold, model.M, model.alpha)
getthreshold(model::sVAEmodel, x) = getthreshold(model.svae, x, model.contamination, model.M, model.alpha; Beta = model.Beta)
params(model::sVAEmodel) = Flux.params(model.svae)

"""
    setthreshold!(model, X)

Set model classification threshold based on given contamination rate.
"""
function setthreshold!(model::sVAEmodel, X)
    model.threshold = getthreshold(model, X)
end

"""
    fit!(model, X)

Trains a sVAEmodel.
"""
function fit!(model::sVAEmodel, X)
    # train the sVAE NN
    fit!(model.svae, X, model.batchsize, model.lambda, M=model.M,
     iterations=model.iterations, cbit = model.cbit, verb = model.verbfit, 
     rdelta = model.rdelta, history = model.history, eta = model.eta)
end

"""
    predict(model, X)

Return estiamted labels.
"""
function predict(model::sVAEmodel, X) 
    return classify(model, X)
end
