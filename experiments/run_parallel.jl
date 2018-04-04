using Distances
using Lazy
using Iterators
using FileIO
using AnomalyDetection

@everywhere begin
	loda_path = "/mnt/output/data/datasets/numerical"
#	loda_path = "./Loda/public/datasets/numerical"
	export_path = "/mnt/output/anomaly" #master path where data will be stored
#	export_path = "./data" #master path where data will be stored
	include(joinpath(Pkg.dir("AnomalyDetection"), "experiments/parallel_utils.jl"))
end

(size(ARGS,1) >0)? repetition = parse(Int64, ARGS[1]) : repetition = 1
datasets = readdir(loda_path)
datasets = filter!(e->(e != "url"), datasets)
datasets = filter!(e->(e != "anomalySets.r"), datasets)
datasets = filter!(e->(e != ".DS_Store"), datasets)
dpaths = joinpath.(export_path, datasets)
pmap(i -> i[1](i[2][1],i[2][2],i[3]),product([trainAE, trainVAE, trainsVAE, trainGAN, 
	trainfmGAN, trainkNN], [(dpaths[n], datasets[n]) for n in size(dpaths,1)], 1:repetition))
