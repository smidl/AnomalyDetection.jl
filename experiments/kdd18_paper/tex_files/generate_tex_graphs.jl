(length(ARGS) > 0)? (contains(ARGS[1], "v")? verb=true:verb=false ) : verb = false
(length(ARGS) > 0)? (contains(ARGS[1], "r")? recomputedata=true:recomputedata=false ) : recomputedata = false

if recomputedata
	ARGS = []
	include("../rank_algorithms.jl")
else
	ARGS = ["d"]
	# load this to obtain the proper paths
	include("../evaluate_experiment.jl")
	# then load the apropriate files
	rankeddf = loadtable(joinpath(evalpath, "ranksummary.csv"), 2);
end

# path where the paper is
outpath = "/home/vit/Dropbox/Cisco/kdd2018/text/misc"
mkpath(outpath)

_algnames = ["kNN", "IForest", "AE", "VAE", "GAN", "fmGAN"]

#cv = 1.1578 # Nemenyi 0.1 critical value
cv = 1.2746 # Nemenyi 0.05 critical value

tacds = ranks2tikzcd(convert(Array, rankeddf[1,2:end]), _algnames, cv, "Critical difference diagram for the first hyperparameter selection criterion.", label = "fig:cdd1")
tracds = ranks2tikzcd(convert(Array, rankeddf[2,2:end]), _algnames, cv, "Critical difference diagram for the second hyperparameter selection criterion.", label = "fig:cdd2")
t5acds = ranks2tikzcd(convert(Array, rankeddf[3,2:end]), _algnames, cv, "Critical difference diagram for the third hyperparameter selection criterion at 5\\% most anomalous samples.", label = "fig:cdd3")
t1acds = ranks2tikzcd(convert(Array, rankeddf[4,2:end]), _algnames, cv, "Critical difference diagram for the third hyperparameter selection criterion at 1\\% most anomalous samples.", label = "fig:cdd4")

# output to txt
string2file(joinpath(outpath, "cdd1.tex"), tacds)
string2file(joinpath(outpath, "cdd2.tex"), tracds)
string2file(joinpath(outpath, "cdd3.tex"), t5acds)
string2file(joinpath(outpath, "cdd4.tex"), t1acds)
