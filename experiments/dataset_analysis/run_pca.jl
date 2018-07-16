# this script transforms all the datasets in experiments/datasets using a 2D PCA and save them 
# to the outfolder specified as the second script calling argument

include("pca_utils.jl")
using ProgressMeter

if length(ARGS) < 1
	error("Specify the outpath in calling arguments, eg. 'julia run_pca.jl path'")
end

outpath = ARGS[1]
inpath = joinpath(Pkg.dir("AnomalyDetection"), "experiments/datasets")
dirs = readdir(inpath)

p = Progress(length(dirs),0.1)

for dir in dirs
	indir = joinpath(inpath, dir)
	outdir = joinpath(outpath, dir)
	data2Dpca(indir, outdir)
	#println("Dataset $dir processed")	  
	next!(p)
end