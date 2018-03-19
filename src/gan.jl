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
    pz # code distribution
end


"""
	GAN(generator, discriminator)

Basic GAN constructor.
"""
GAN(G::Flux.Chain, D::Flux.Chain; pz=rand) = GAN(G, freeze(G), D, freeze(D), pz)

"""
	GAN(gsize, dsize, [pz, activation])

Constructor for the GAN object. 
gsize - vector of Ints describing generator layers sizes
dsize - vector of Ints describing discriminator layers sizes, including the last scalar layer 
pz - code distribution
activation - activation function common to all layers
"""
function GAN(gsize, dsize; pz = rand, activation = Flux.relu)
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
	for i in 3:(size(dsize,1)-1)
		discriminator = Chain(discriminator, Dense(dsize[i-1], dsize[i], activation))
	end
    discriminator = Chain(discriminator, Dense(dsize[end-1], dsize[end], σ))

	return GAN(generator, discriminator, pz=pz)
end


### auxilliary functions ###

"""
	generate_sample(gan)

Generate one sample.
"""
generate_sample(gan::GAN) = gan.g(gan.pz(Int(size(gan.g.layers[1].W,2)))).data

"""
	generate_sample(gan, n)

Generate n samples.
"""
generate_sample(gan::GAN, n::Int) = gan.g(gan.pz(Int(size(gan.g.layers[1].W,2)), n)).data

### training ###
"""
	Dloss(gan, X, Z)

Discriminator loss.
"""
Dloss(gan::GAN, X, Z) = - 0.5*(mean(log.(gan.d(X))) + mean(log.(1 - gan.d(gan.gg(Z)))))

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
	fit!(gan, X, [iterations throttleit, verb])

Trains the GAN.
"""
function fit!(gan::GAN, X, M; iterations=1000, throttleit = 200, verb = true)
	# settings
	Dopt = ADAM(params(gan.d))
	Gopt = ADAM(params(gan.g))
	
	# problem size
	N = size(X,2)
	zdim = size(params(model.g)[1],2)

	# train the GAN
	for i in 1:iterations
		# sample data and generate codes
		x = X[:,sample(1:N, M, replace=false)]
		z = gan.pz(zdim, M)
                
        Flux.train!(Dloss, [(gan, x, z)], Dopt)
		Flux.train!(Gloss, [(gan, z)], Gopt)
	
        
		# callback
		if verb && i%throttleit==0
			evalloss(model, x, z)
		end
	end
end
