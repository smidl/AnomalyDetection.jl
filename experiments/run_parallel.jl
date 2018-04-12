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
runexperiment(datasets[1], 3, "kNN")
runexperiment(datasets[2], 2, "AE")
runexperiment(datasets[3], 1, "VAE")

#pmap(i -> i[1](joinpath(export_path,i[2]),i[2],i[3]),product([trainAE, trainVAE, trainsVAE, trainGAN, 
#	trainfmGAN, trainkNN], datasets, repetition))
