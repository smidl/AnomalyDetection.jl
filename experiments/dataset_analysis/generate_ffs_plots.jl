# run as 'julia generate_ffs_plots.jl outpath top nlines [repetitions]'
# outpath - where plots will be stored
# top = [vae,knn]
# nlines = number of top different results to process
# repetitions = multiple runs with different random seed for data splitting
fpath = @__DIR__
include(joinpath(fpath,"utils.jl"))
using ProgressMeter

# get args
outpath = ARGS[1]
top = ARGS[2]
#nlines = Int(parse(ARGS[3]))
nlines = ARGS[3]
reps = ((length(ARGS)>3)? Int(parse(ARGS[4])) : 1)

# get the top df
variant = "some"
ARGS = [variant, top, nlines]
println("These findfeatures results will be processed and plotted:")
include(joinpath(fpath,"show_ffs_res.jl"))
println("")

# make the graphs
p = Progress(nlines*reps,0.5)
for rep in 1:reps
	for i in 1:nlines
		plot_ffs_all(df, i, variant, outpath; seed = rep)
		close() # so that there are no excess plots
		ProgressMeter.next!(p)
	end
end