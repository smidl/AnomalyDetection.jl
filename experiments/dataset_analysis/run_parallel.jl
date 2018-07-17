# version for experiments run on AWS
using Distances
using Lazy
using IterTools
using FileIO
using AnomalyDetection
using DataStructures

@everywhere begin
	loda_path = "pca_2D-data"
	export_path = "pca_2D-experiment" #master path where data will be stored
	include("../kdd18_paper/parallel_utils.jl")
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1
nhdims = (size(ARGS,1) >0) ? parse(Int64, ARGS[2]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistant-connection"]))

pmap(x -> runexperiments(x[1], iteration, x[2], nhdims),
	product(datasets, ["kNN", "AE", "VAE", "GAN", "fmGAN"]))
