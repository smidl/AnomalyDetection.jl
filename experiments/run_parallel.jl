using Distances
using Lazy
using IterTools
using FileIO
using AnomalyDetection
using DataStructures

@everywhere begin
	loda_path = "/mnt/output/data/datasets/numerical"
	export_path = "/mnt/output/anomaly" #master path where data will be stored
	include(joinpath(Pkg.dir("AnomalyDetection"), "experiments/parallel_utils.jl"))
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(s -> s != "url") 

#pmap(x -> runexperiment(x[1], x[3], x[2]), 
#	product(datasets, ["kNN", "AE", "VAE", "sVAE", "GAN", "fmGAN"], iteration))

pmap(x -> runexperiment(x[2], x[3], x[1]), 
	product(["kNN", "AE", "VAE", "sVAE", "GAN", "fmGAN"], datasets, iteration))