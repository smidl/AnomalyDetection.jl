using IterTools

master_path = ARGS[1]


@everywhere begin
	include("../eval.jl")

	# where is the data (extracted or original) stored
	#data_path = "/opt/output/extracted"
	#data_path = "/home/vit/vyzkum/anomaly_detection/data/aws/anomaly"
	#data_path = "/home/vit/vyzkum/anomaly_detection/data/aws/extracted"
	#data_path = "/home/vit/vyzkum/anomaly_detection/data/aws_final/extracted"
	data_path = joinpath(master_path, "extracted")

	# where the individual experimen results will be stored
	#outpath = "/opt/output/output"
	#outpath = "/home/vit/vyzkum/anomaly_detection/data/aws/output"
	#outpath = "/home/vit/vyzkum/anomaly_detection/data/aws/ex_output"
	#outpath = "/home/vit/vyzkum/anomaly_detection/data/aws_final/output"
	outpath = joinpath(master_path, "output")
	
	# where the summary tables will be stored
	#evalpath = "/opt/output/eval"
	#evalpath = "/home/vit/vyzkum/anomaly_detection/data/aws/eval"
	#evalpath = "/home/vit/vyzkum/anomaly_detection/data/aws/ex_eval"
	#evalpath = "/home/vit/vyzkum/anomaly_detection/data/aws_final/eval"
	evalpath = joinpath(master_path, "eval")
end

datasets = filter(s->s!="persistant-connection",readdir(data_path))
datasets = filter(s->s!="gisette", datasets)
mkpath(outpath)
mkpath(evalpath)

#algnames = ["kNN", "kNNPCA", "IsoForest", "AE", "VAE", "sVAE", "GAN", "fmGAN"]
algnames = ["kNN", "IsoForest", "AE", "VAE", "GAN", "fmGAN"]
#algnames = ["kNN", "kNNPCA"]

if (length(ARGS) > 1)
	if ARGS[2] == "d"
		dryrun = true
	else
		dryrun = false
	end
else
	dryrun = false
end

if !dryrun
	# create a dataframe with a row for each experiment
	println("Computing basic experiment statistics...")
	map(x->datastats(x[1], x[2], data_path, outpath), 
		product(datasets, algnames))
	println("Done.")
end
