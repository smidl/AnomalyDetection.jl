# run as 'julia findfeatures.jl 10 s'
# s stands for some data - only anomalies of one difficulty level are sampled
using EvalCurves, AnomalyDetection
fpath = @__DIR__
include(joinpath(fpath, "ffs_util.jl"))

# paths
datapath = joinpath(fpath, "../datasets")
outpath = joinpath(fpath, "ffs-experiment")
datasets = readdir(datapath)

# arguments
la = length(ARGS)
maxtries = ((la > 0)?  Int64(parse(ARGS[1])) : 10)
alldata = ((la > 1)? ((ARGS[2]=="s")? false : true) : true)
outpath = joinpath(outpath,"$(maxtries)_$alldata")
mkpath(outpath)

# run loop over datasets
for dataset in datasets
	println("Processing $dataset...")
	df = scorefeatures(dataset, maxtries, alldata)
	f = joinpath(outpath, "$dataset.csv")
	CSV.write(f, df)
	showall(df)
end