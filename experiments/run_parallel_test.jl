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

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(s -> s != "url") filter(x -> x != "gisette")

#@time runexperiment(datasets[1], 3, "kNN")
#@time runexperiment(datasets[2], 2, "AE")
#@time runexperiment(datasets[3], 1, "VAE")
#@time runexperiment(datasets[4], 1, "sVAE")
#@time runexperiment(datasets[5], 4, "GAN")
#@time runexperiment(datasets[6], 5, "fmGAN")


pmap(x -> runexperiment(x[1], x[3], x[2]), 
	#product(datasets, ["VAE", "kNN", "AE", "GAN", "VAE", "sVAE", "fmGAN"], iteration))
	product(datasets, ["GAN"], 1:10))

#pmap(x -> runexperiment(x[2], x[3], x[1]), 
#	product(["kNN", "AE", "VAE", "sVAE", "GAN", "fmGAN"], datasets, iteration))