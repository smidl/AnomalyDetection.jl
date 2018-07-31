# run as 'julia generate_tsne_plots.jl outpath [reps]'
# outpath - where plots will be stored
# reps = number of data splits to be tested

fp = @__DIR__
include(joinpath(fp,"utils.jl"))
using ProgressMeter
inpath = joinpath(fp,"tsne_2D-data")

# get args
outpath = ARGS[1]
datasets = readdir(inpath)
reps = ((length(ARGS)>1)? Int(parse(ARGS[2])) : 1)

# make the graphs
p = Progress(length(datasets)*reps,0.5)

for dataset in datasets
	for rep in 1:reps
		plot_general_all(dataset, inpath, "tSne", outpath; seed = rep)
		close() # so that there are no excess plots
		ProgressMeter.next!(p; showvalues = [(:dataset, dataset), (:rep, rep)])
	end
end