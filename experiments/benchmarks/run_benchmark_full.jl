using Distances
using Lazy
using IterTools
using FileIO
using AnomalyDetection
using DataStructures

@everywhere begin
#	algorithms = ["kNN", "AE", "VAE", "sigmaVAE", "GAN", "fmGAN", "sVAE"]
	algorithms = ["kNN", "AE", "VAE", "GAN", "fmGAN"]
	#algorithms = ["AE", "VAE", "GAN", "fmGAN"]
	#algorithms = ["VAEensemble"]
#	algorithms = ["IsoForest"]

	if "IsoForest" in algorithms
		println("For Isolation Forest, paralell run is not implemented. Run without the -p flag.")
		isoforest = true
		include("../benchmarks/isolation_forest.jl")
	else
		isoforest = false
	end

	loda_path = "../dataset_analysis/tsne_2D-data"
	host = gethostname()
	#master path where data will be stored
	if host == "vit"
		export_path = "/home/vit/vyzkum/anomaly_detection/data/benchmarks/tsne_2D-all-minlat2/data" 
	elseif host == "axolotl.utia.cas.cz"
		export_path = "/home/skvara/work/anomaly_detection/data/benchmarks/tsne_2D-all-minlat2/data"
	end
	include("parallel_utils.jl")
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1
nhdims = (size(ARGS,1) >0) ? parse(Int64, ARGS[2]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistent-connection"]))

pmap(x -> runexperiments(x[1], iteration, x[2], nhdims),
	product(datasets, algorithms))
