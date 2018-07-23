using EvalCurves, AnomalyDetection
fpath = @__DIR__
include(joinpath(fpath, "ffs_util.jl"))

# paths
datapath = joinpath(fpath, "../datasets")
outpath = joinpath(fpath, "ffs-experiment")
mkpath(outpath)
datasets = readdir(datapath)

# run loop over datasets
for dataset in datasets
	println("Processing $dataset...")
	df = scorefeatures(dataset)
	f = joinpath(outpath, "$dataset.csv")
	CSV.write(f, df)
	showall(df)
end