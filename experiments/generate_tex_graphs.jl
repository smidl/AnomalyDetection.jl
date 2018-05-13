(length(ARGS) > 0)? ((ARGS[1] == "v")? verb=true:verb=false ) : verb = false

ARGS = []
include("rank_algorithms.jl")

outpath = "./tex_graphs"
mkpath(outpath)

_algnames = ["kNN", "IForest", "AE", "VAE", "GAN", "fmGAN"]

c = 1.1578
tacds = ranks2tikzcd(convert(Array, rankeddf[1,2:end]), _algnames, c, "Critical diagram for the first hyperparameter selection criteria.", label = "fig:cd1")
tracds = ranks2tikzcd(convert(Array, rankeddf[2,2:end]), _algnames, c, "Critical diagram for the second hyperparameter selection criteria.", label = "fig:cd2")
t1acds = ranks2tikzcd(convert(Array, rankeddf[3,2:end]), _algnames, c, "Critical diagram for the third hyperparameter selection criteria at 1\\% most anomalous samples.", label = "fig:cd3")
t5acds = ranks2tikzcd(convert(Array, rankeddf[4,2:end]), _algnames, c, "Critical diagram for the third hyperparameter selection criteria at 1\\% most anomalous samples.", label = "fig:cd4")

# output to txt
string2file(joinpath(outpath, "tacds.txt"), tacds)
string2file(joinpath(outpath, "tracds.txt"), tracds)
string2file(joinpath(outpath, "t1acds.txt"), t1acds)
string2file(joinpath(outpath, "t5acds.txt"), t5acds)

