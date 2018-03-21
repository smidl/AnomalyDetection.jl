module AnomalyDetection

#using StatsBase, Distances
using Distances
import ScikitLearn: @sk_import, fit!, predict
using Flux
import Base.Iterators.repeated


export Basicset, Dataset, VAE, VAEmodel, AE, AEmodel, GAN, GANmodel

include("vae.jl")
include("ae.jl")
include("gan.jl")
include("utils.jl")

end