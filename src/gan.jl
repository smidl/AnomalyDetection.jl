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


### auxilliary functions ###

"""
	generate(gan)

Generate one sample.
"""
generate(gan::GAN) = gan.g(gan.pz(size(params(gan.g)[1],2))).data

"""
	generate(gan, n)

Generate n samples.
"""
generate(gan::GAN, n::Int) = gan.g(gan.pz(size(params(gan.g)[1],2), n)).data

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
Gloss(gan::GAN, Z) = - mean(log.(gan.dd(gan.g(Z))))

"""
	rerr(gan, X, Z)

Crude estimate of reconstruction error.
"""
rerr(gan::GAN, X, Z) = abs(mean(gan.g(Z).data) - mean(X))

"""
	evalloss(gan, X, Z)
"""
evalloss(gan::GAN, X, Z) = print("discriminator loss: ", Dloss(gan, X, Z).data,  
	"\ngenerator loss: ", Gloss(gan, Z).data, 
	"\nreconstruction error: ", rerr(gan, X, Z), "\n\n")

"""
	fit!(gan, X, [iterations, cbit, verb, rdelta])

Trains a GAN.

gan - struct of type GAN
X - data array with instances as columns
M - number of samples to be selected from X and sampled from pz
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
"""
function fit!(gan::GAN, X, M; iterations=1000, cbit = 200, verb = true, rdelta = Inf)
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
		x = X[:,sample(1:N, M, replace=false)]
		z = gan.pz(zdim, M)
                
        # discriminator training
        Dl = Dloss(x,z)
        Flux.Tracker.back!(Dl)
        Dopt()
		
		# generator training	
        Gl = Gloss(z)
        Flux.Tracker.back!(Gl)
        Gopt()
	
		# callback
		if verb && i%cbit==0
			evalloss(gan, x, z)
		end

		# if a stopping condition is present
		if rdelta < Inf
			re = rerr(model, x, z) 
			if re < rdelta
				println("Training ended prematurely after $i iterations,
					\nreconstruction error $re < $rdelta")
				break
			end
		end
	end
end

