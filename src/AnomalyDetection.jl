module AnomalyDetection

#using StatsBase, Distances
using Distances
import ScikitLearn: @sk_import, fit!, predict
using Flux
import Base.Iterators.repeated
using PyPlot
import PyPlot.plot # so we can add new methods to plot()

export Basicset, Dataset, VAE, VAEmodel, AE, AEmodel, GAN, GANmodel, 
	sVAE, sVAEmodel, fmGAN, fmGANmodel 

include("vae.jl")
include("ae.jl")
include("svae.jl")
include("gan.jl")
include("fmgan.jl")
include("utils.jl")

end