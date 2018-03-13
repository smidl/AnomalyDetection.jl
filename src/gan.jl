"""
	freeze(m)

Creates a non-trainable copy of a Flux object.
"""
freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)

"""
	GAN

Flux-like object representing a generative adversarial network.
"""
struct GAN
	g # generator
	gg # non-trainable generator copy
	d # discriminator
	dd # non-trainable discriminator copy
end

"""
	GAN(generator, discriminator)

Basic GAN constructor.
"""
GAN(G::Flux.Chain, D::Flux.Chain) = GAN(G, freeze(G), D, freeze(D))

"""
	GAN(gsize, dsize, [activation])

Constructor for the GAN object. 
gsize - vector of Ints describing generator layers sizes
dsize - vector of Ints describing discriminator layers sizes, including the last scalar layer 
activation - activation function common to all layers
"""
function GAN(gsize, dsize; activation = Flux.relu)
	@assert size(gsize,1) >= 3
	@assert size(dsize,1) >= 3
	@assert dsize[end] == 1
	@assert gsize[end] == dsize[1]

	# generator
	generator = Dense(gsize[1], gsize[2], activation)
	for i in 3:size(gsize,1)
		generator = Chain(generator, Dense(gsize[i-1], gsize[i], activation))
	end

	# discriminator
	discriminator = Dense(dsize[1], dsize[2], activation)
	for i in 3:size(dsize,1)
		discriminator = Chain(discriminator, Dense(dsize[i-1], dsize[i], activation))
	end

	return GAN(generator, discriminator)
end

### auxilliary functions ###

"""
	generate_sample(gan)

Generate one sample using uniformly distributed code.
"""
generate_sample(gan::GAN) = gan.g(rand(Int(size(gan.g.layers[1].W,2)))).data

"""
	generate_sample(gan, n)

Generate n samples using uniformly distributed code.
"""
generate_sample(gan::GAN, n::Int) = gan.g(rand(Int(size(gan.g.layers[1].W,2)), n)).data

### training ###
"""
	Dloss(gan, X, Z)

Discriminator loss.
"""
Dloss(gan::GAN, X, Z) = - mean(log.(gan.d(X))) - mean(log.(1 - gan.d(gan.gg(Z))))

"""
	Gloss(gan, Z)

Generator loss.
"""
Gloss(gan::GAN, Z) = - mean(log.(gan.dd(gan.g(X))))

"""
	evalloss(gan, X, Z)
"""
evalloss(gan::GAN, X, Z) = print("discriminator loss: ", Dloss(gan, X, Z).data[1], 
	"\ngenerator loss: ", Gloss(gan, Z).data[1], 
	"\nreconstruction error: ", Flux.mse(gan.g(Z), X).data[1], "\n\n")

"""
	fit!(gan, X; iterations=1000, throttle = 5, verb = true)

Trains the GAN.
"""
function fit!(gan::GAN, X, M; iterations=1000, throttle = 5, verb = true)
	# settings
	Dopt = ADAM(params(gan.d))
	Gopt = ADAM(params(gan.g))
	
	# problem size
	N = size(X,2)
	zdim = Int(size(gan.g.layers[1].W,2))

	# train the GAN
	for i in 1:iterations
		# sample data and generate codes
		x = X[:,sample(1:N, M, replace=false)]
		z = rand(zdim, M)

		# callback
		evalcb() = evalloss(gan, X, rand(zdim, N))
		cb = Flux.throttle(evalcb, throttle)
		
		if verb
			Flux.train!(Dloss, [(gan, x, z)], Dopt, cb = cb)
			Flux.train!(Gloss, [(gan, z)], Gopt, cb = cb)
		else
			Flux.train!(Dloss, [(gan, x, z)], Dopt)
			Flux.train!(Gloss, [(gan, z)], Gopt)
		end
	end
end
