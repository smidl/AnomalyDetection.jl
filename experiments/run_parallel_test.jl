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

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(s -> s != "url") 
#b = trainkNN(datasets[1], 4);
d = runexperiment(datasets[1], 3, "kNN")
d = runexperiment(datasets[2], 2, "AE")
d = runexperiment(datasets[3], 1, "VAE")
#d = runexperiment(datasets[3], 1, "sVAE")
#c = trainAE(datasets[1],4)
#pmap(i -> i[1](i[2],i[3]),product([trainAE], datasets, 1:iteration))
#pmap(i -> i[1](joinpath(export_path,i[2]),i[2],i[3]),product([trainkNN], datasets, iteration))
