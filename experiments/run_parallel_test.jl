using Distances
using Lazy
using IterTools
using FileIO
push!(LOAD_PATH, "../src")
using AnomalyDetection
using DataStructures

@everywhere begin
	loda_path = "./Loda/public/datasets/numerical"
	export_path = "./data" #master path where data will be stored
	include("parallel_utils.jl")
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistent-connection"])) 

pmap(x -> runexperiment(x[1], x[3], x[2]), 
	#product(datasets, ["VAE", "kNN", "AE", "GAN", "VAE", "sVAE", "fmGAN"], iteration))
	product(datasets, ["GAN"], 1:10))
