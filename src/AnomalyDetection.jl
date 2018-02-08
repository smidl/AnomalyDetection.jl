module AnomalyDetection

#using StatsBase, Distances
using Distances
import ScikitLearn: @sk_import, fit!, predict


export Basicset, Dataset, VAE, VAEmodel

include("vae.jl")
include("utils.jl")

end