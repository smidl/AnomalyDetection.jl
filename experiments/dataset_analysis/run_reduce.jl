# this script transforms all the datasets in experiments/datasets using a 2D PCA/tSNE operation 
# and save them to the outfolder specified as the second script calling argument

include("utils.jl")
using ProgressMeter

if length(ARGS) < 2
	error("Specify the outpath in calling arguments, eg. 'julia run_pca.jl tsne path'")
end

variant = ARGS[1]
outpath = ARGS[2]
inpath = joinpath(Pkg.dir("AnomalyDetection"), "experiments/datasets")
dirs = readdir(inpath)

p = Progress(length(dirs),0.1)

for dir in dirs
	indir = joinpath(inpath, dir)
	outdir = joinpath(outpath, dir)
	println("Processing $dir...")
	dataset2D(indir, outdir, variant)	
	println("$dir processed")  
	if variant == "pca"
		next!(p)
	end
end
println("")