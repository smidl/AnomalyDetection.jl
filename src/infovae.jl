###############################
### infovae NN construction ###
###############################

"""
	infoVAE{encoder, sampler, decoder, variant}

Flux-like structure for the variational autoencoder.
"""
struct infoVAE{E, S, D, P, V<:Val}
	encoder::E
	sampler::S
	decoder::D
	pz::P
	variant::V
end

# and make it trainable
Flux.@treelike infoVAE

infoVAE(E,S,D,P,V::Symbol = :unit) = infoVAE(E,S,D,P,Val(V))

# make the struct callable
(ivae::infoVAE)(X) = ivae.decoder(ivae.sampler(ivae.encoder(X)))

"""
	infoVAE(esize, dsize; [activation, layer, variant, pz])

Initialize an infoVAE model with given encoder size and decoder size.

esize - vector of ints specifying the width anf number of layers of the encoder
\ndsize - size of decoder
\nactivation [Flux.relu] - arbitrary activation function
\nlayer [Flux.Dense] - type of layer
\nvariant [:unit] - :unit - output has unit variance
\n 		          - :sigma - the variance of the output is estimated
\npz [randn] - what is pz
"""
function infoVAE(esize::Array{Int64,1}, dsize::Array{Int64,1}; activation = Flux.relu,
		layer = Flux.Dense, variant = :unit, pz = randn)
	@assert variant in [:unit, :sigma]
	@assert size(esize, 1) >= 3
	@assert size(dsize, 1) >= 3
	@assert esize[end] == 2*dsize[1]
	(variant==:unit) ? (@assert esize[1] == dsize[end]) :
		(@assert esize[1]*2 == dsize[end])

	# construct the encoder
	encoder = aelayerbuilder(esize, activation, layer)

	# construct the decoder
	decoder = aelayerbuilder(dsize, activation, layer)

	# finally construct the ae struct
	ivae = infoVAE(encoder, samplenormal, decoder, variant)

	return ivae
end

################
### training ###
################

"""
	likelihood(infovae, X)

Likelihood of an autoencoded sample X.
"""
function likelihood(ivae::infoVAE{E,S,D,V}, X) where {E,S,D,V<:Val{:sigma}}
	vx = ivae(X)
	μ, σ2 = mu(vx), sigma2(vx)
	return likelihood(X,μ, σ2)
end
function likelihood(ivae::infoVAE{E,S,D,V}, X) where {E,S,D,V<:Val{:unit}}
	μ = ivae(X)
	return likelihood(X,μ)
end

"""
	likelihood(infovae, X, M)

Likelihood of an autoencoded sample X sampled M times.
"""
likelihood(ivae::infoVAE, X, M) = mean([likelihood(ivae, X) for m in 1:M])

"""
	KL(infovae, X)

KL divergence between the encoder output and unit gaussian.
"""
function KL(ivae::infoVAE, X) 
	ex = ivae.encoder(X)
	KL(mu(ex), sigma2(ex))
end

"""
	MMD(infovae, X, σ)

MMD of the infoVAE encoder and a given pz.
"""
MMD(ivae::infoVAE, X, σ) = MMD(X,
	x -> samplenormal(ivae.encoder(x)),
	x -> ivae.pz(Int(size(ivae.encoder.layers[end].W,1)/2), size(X,2)),
	σ)

"""
	loss(infovae, X, M, λ, α, σ)

Loss function of the infoVAE. M is the number of samples taken in the first term of
\nloss = - E[log(p(x|z))] + (1-α)KLD(q(z|x)||p(z)) + (α + λ - 1)MMD_σ(q(z)||p(z)).
"""
loss(ivae::infoVAE, X, M, λ, α, σ) = 
	-likelihood(ivae,X,M) + Float(1-α)*KL(ivae, X) + Float(α + λ - 1)*MMD(ivae, X, σ)

"""
	getlosses(infovae, X, M, λ, α, σ)

Return the numeric values of current losses of an infoVAE.
"""
getlosses(ivae::infoVAE, X, M, λ, α, σ) = (
	Flux.Tracker.data(loss(ivae, X, M, λ, α, σ)),
	Flux.Tracker.data(-likelihood(ivae,X,M)),
	Flux.Tracker.data(KL(ivae, X)),
	Flux.Tracker.data(MMD(ivae, X, σ))
	)

"""
	evalloss(infovae, X, M, λ, α, σ)

Print infoVAE loss function values.
"""
function evalloss(ivae::infoVAE, X, M, λ, α, σ) 
	l, lk, kl, mmd = getlosses(ivae, X, M, λ, α, σ)
	print("loss: ", l,
	"\nlikelihood: ", lk,
	"\nKL: ", kl, 
	"\nMMD: ", mmd, 
	"\n\n")
end

