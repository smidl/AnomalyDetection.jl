module AnomalyDetection
import Base.convert
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, 
	false_negative_rate
using StatsBase: sample
using Adapt, FluxExtensions, Distances, Flux, MultivariateStats
using ValueHistories, ProgressMeter, DelimitedFiles, Statistics, Random
using Pkg
import Base.Iterators.repeated
import Flux: params
import Base.cat

# Float 32 is almost 2x faster
const Float = Float32

export Basicset, Dataset, VAE, VAEmodel, AE, AEmodel, GAN, GANmodel, 
	sVAE, sVAEmodel, fmGAN, fmGANmodel, kNN, Ensemble

# generative model abstract type
abstract type genmodel
end

include("ae.jl")
include("vae.jl")
include("svae.jl")
include("gan.jl")
include("fmgan.jl")
include("knn.jl")
include("utils.jl")
include("samplers.jl")
include("ensembles.jl")

end