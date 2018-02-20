module AnomalyDetection

#using StatsBase, Distances
using Distances
import ScikitLearn: @sk_import, fit!, predict


export Basicset, Dataset, VAE, VAEmodel, AE, AEmodel

include("vae.jl")
include("ae.jl")
include("utils.jl")

end