module AnomalyDetection

import Base.convert
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, 
	false_negative_rate
using StatsBase: sample
using Adapt, Distances, Flux, MultivariateStats
import Adapt: adapt
import Base.Iterators.repeated
#using PyPlot
#import PyPlot.plot # so we can add new methods to plot()

# Float 32 is almost 2x faster
const Float = Float32

export Basicset, Dataset, VAE, VAEmodel, AE, AEmodel, GAN, GANmodel, 
	sVAE, sVAEmodel, fmGAN, fmGANmodel, kNN

include("ae.jl")
include("vae.jl")
include("svae.jl")
include("gan.jl")
include("fmgan.jl")
include("knn.jl")
include("utils.jl")

end