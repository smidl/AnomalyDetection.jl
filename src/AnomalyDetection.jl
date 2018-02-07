module AnomalyDetection

#using StatsBase, Distances
using Distances
import ScikitLearn: @sk_import, fit!, predict


export Basicset, Dataset, VAE

include("utils.jl")
include("vae.jl")

end