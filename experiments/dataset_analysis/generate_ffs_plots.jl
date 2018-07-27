# run as 'julia generate_ffs_plots.jl outpath top nlines'
# outpath - where plots will be stored
# top = [vae,knn]
# nlines = number of top different results to process

fp = @__DIR__
include(joinpath(fp,"utils.jl"))
using ProgressMeter

# get args
outpath = ARGS[1]
top = ARGS[2]
#nlines = Int(parse(ARGS[3]))
nlines = ARGS[3]

# get the top df
variant = "some"
ARGS = [variant, top, nlines]
println("These findfeatures results will be processed and plotted:")
include(joinpath(fp,"show_ffs_res.jl"))
println("")

# make the graphs
p = Progress(nlines,0.5)
for i in 1:nlines
	plot_ffs_all(df, i, variant, outpath)
	ProgressMeter.next!(p)
end