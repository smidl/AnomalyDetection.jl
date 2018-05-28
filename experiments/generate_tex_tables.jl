(length(ARGS) > 0)? (contains(ARGS[1], "v")? verb=true:verb=false ) : verb = false
(length(ARGS) > 0)? (contains(ARGS[1], "r")? recomputedata=true:recomputedata=false ) : recomputedata = false

if recomputedata
	ARGS = []
	include("rank_algorithms.jl")
else
	ARGS = ["d"]
	# load this to obtain the proper paths
	include("evaluate_experiment.jl")
	# then load the apropriate files
	testauc = loadtable(joinpath(evalpath, "testauc.csv"), 2)
	ranktestauc = loadtable(joinpath(evalpath, "ranktestauc.csv"), 2)
	trainauc = loadtable(joinpath(evalpath, "trainauc.csv"), 2);
	ranktrainauc = loadtable(joinpath(evalpath, "ranktrainauc.csv"), 2);
	top5auc = loadtable(joinpath(evalpath, "top5auc.csv"), 2);
	ranktop5auc = loadtable(joinpath(evalpath, "ranktop5auc.csv"), 2);
	top1auc = loadtable(joinpath(evalpath, "top5auc.csv"), 2);
	ranktop1auc = loadtable(joinpath(evalpath, "ranktop5auc.csv"), 2);
	meanfitt = loadtable(joinpath(evalpath, "meanfitt.csv"), 2);
	rankmeanfitt = loadtable(joinpath(evalpath, "rankmeanfitt.csv"), 2);
	meanpredictt = loadtable(joinpath(evalpath, "meanpredictt.csv"), 2);
	rankmeanpredictt = loadtable(joinpath(evalpath, "rankmeanpredictt.csv"), 2);
	testaauc = loadtable(joinpath(evalpath, "testaauc.csv"), 2)
	ranktestaauc = loadtable(joinpath(evalpath, "ranktestaauc.csv"), 2)
	trainaauc = loadtable(joinpath(evalpath, "trainaauc.csv"), 2);
	ranktrainaauc = loadtable(joinpath(evalpath, "ranktrainaauc.csv"), 2);
	top5aauc = loadtable(joinpath(evalpath, "top5aauc.csv"), 2);
	ranktop5aauc = loadtable(joinpath(evalpath, "ranktop5aauc.csv"), 2);
	top1aauc = loadtable(joinpath(evalpath, "top5aauc.csv"), 2);
	ranktop1aauc = loadtable(joinpath(evalpath, "ranktop5aauc.csv"), 2);
	valuedf = loadtable(joinpath(evalpath, "valuesummary.csv"), 2);
	rankeddf = loadtable(joinpath(evalpath, "ranksummary.csv"), 2);
end

outpath = "/home/vit/Dropbox/Cisco/kdd2018/text/misc"
mkpath(outpath)

 
const shortnames = ["aba", "blo", "brc", "brt", "car", "eco", "gla", "hab", "ion", "iri", 
	"iso", "let", "lib", "mad", "mag", "min", "mul", "mus", "pag", "par", "pen", "pim", "son", 
	"spe", "ssa", "sse", "ssh", "sve", "syn", "ver", "wal", "wa1", "wa2", "win", "yea", "avg"]

#(size(ARGS,1)>1)? v = ARGS[1] : v = "verb"

# table 2 - mean ranks on normal auroc
rankaurocsum = rounddf(rankeddf[1:4,:], 2, 2)
rankaurocsum[:test][3] = "top 5\\%"
rankaurocsum[:test][4] = "top 1\\%"
rename!(rankaurocsum, :IsoForest, :IForest)
rename!(rankaurocsum, :test, Symbol(" "))
rankaurocsum = rpaddf(rankaurocsum, 2)
sra = df2tex(rankaurocsum, "Average ranks of algorithms for different hyperparameter selection criteria.",
	label = "tab:aucsummary",
	firstvline = true)

# table 3 - mean fit/predict times
meantimessum = rounddf(valuedf[5:6,:],2,2)
meantimessum[:test][1] = "\$t_f\$ [s]"
meantimessum[:test][2] = "\$t_p\$ [s]"
rename!(meantimessum, :IsoForest, :IForest)
rename!(meantimessum, :test, Symbol(" "))
meantimessum = rpaddf(meantimessum, 2)
smt = df2tex(meantimessum, "Average fit \$t_f\$ and predict \$t_p\$ times.",
	label = "tab:timesummary",
	firstvline = true)

# table 4 - mean ranks on augmented auroc
rankaaurocsum = rounddf(rankeddf[7:10,:], 2, 2)
rankaaurocsum[:test][1] = "test auc"
rankaaurocsum[:test][2] = "train auc"
rankaaurocsum[:test][3] = "top 5\\%"
rankaaurocsum[:test][4] = "top 1\\%"
rename!(rankaaurocsum, :IsoForest, :IForest)
rename!(rankaaurocsum, :test, Symbol(" "))
rankaaurocsum = rpaddf(rankaaurocsum, 2)
sraa = df2tex(rankaaurocsum, 
	"Average ranks of algorithms for different hyperparameter selection criteria using the augmented AUROC.",
	label = "tab:aaucsummary",
	firstvline = true)

# table 5 - mean scores of the first test with ranks
testaurocdf = miss2hyphen!(rounddf(testauc,2,2))
row = rounddf(valuedf[1,:],2,2)
rename!(row, :test, :dataset)
testaurocdf = [testaurocdf; row]
rename!(testaurocdf, :IsoForest, :IForest)
testaurocdf = rpaddf(testaurocdf, 2)

ranktestaurocdf = miss2hyphen!(rounddf(ranktestauc,2,2))
tsadf = mergedfs(testaurocdf, ranktestaurocdf)
tsadf[:dataset] = shortnames
tsas = df2tex(tsadf,
	"AUROC scores and ranks of algorithms using the first hyperparameter selection criterion. The last line is an average.",
	label = "tab:testaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 6 - mean scores of the second test with ranks
trainaurocdf = miss2hyphen!(rounddf(trainauc,2,2))
row = rounddf(valuedf[2,:],2,2)
rename!(row, :test, :dataset)
trainaurocdf = [trainaurocdf; row]
rename!(trainaurocdf, :IsoForest, :IForest)
trainaurocdf = rpaddf(trainaurocdf, 2)

ranktrainaurocdf = miss2hyphen!(rounddf(ranktrainauc,2,2))
tadf = mergedfs(trainaurocdf, ranktrainaurocdf)
tadf[:dataset] = shortnames
tas = df2tex(tadf,
	"AUROC scores and ranks of algorithms using the second hyperparameter selection criterion.",
	label = "tab:trainaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 7 - mean scores of the third 1% test with ranks
top1aurocdf = miss2hyphen!(rounddf(top1auc,2,2))
row = rounddf(valuedf[3,:],2,2)
rename!(row, :test, :dataset)
top1aurocdf = [top1aurocdf; row]
rename!(top1aurocdf, :IsoForest, :IForest)
top1aurocdf = rpaddf(top1aurocdf, 2)

ranktop1aurocdf = miss2hyphen!(rounddf(ranktop1auc,2,2))
t1adf = mergedfs(top1aurocdf, ranktop1aurocdf)
t1adf[:dataset] = shortnames
t1as = df2tex(t1adf,
	"AUROC scores and ranks of algorithms using the third hyperparameter selection criterion, using top 1\\% of samples.",
	label = "tab:top1aucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 8 - mean scores of the third 5% test with ranks
top5aurocdf = miss2hyphen!(rounddf(top5auc,2,2))
row = rounddf(valuedf[4,:],2,2)
rename!(row, :test, :dataset)
top5aurocdf = [top5aurocdf; row]
rename!(top5aurocdf, :IsoForest, :IForest)
top5aurocdf = rpaddf(top5aurocdf, 2)

ranktop5aurocdf = miss2hyphen!(rounddf(ranktop5auc,2,2))
t5adf = mergedfs(top5aurocdf, ranktop5aurocdf)
t5adf[:dataset] = shortnames
t5as = df2tex(t5adf,
	"AUROC scores and ranks of algorithms using the third hyperparameter selection criterion, using top 5\\% of samples.",
	label = "tab:top5aucfull",
	fitcolumn = true, lasthline = true, firstvline = true)


if verb
	# output
	println("\nTable 2:\n\n",sra,"\n")
	println("\nTable 3:\n\n",smt,"\n")
	println("\nTable 4:\n\n",sraa,"\n")
	println("\nTable 5:\n\n",tsas,"\n")
	println("\nTable 6:\n\n",tas,"\n")
	println("\nTable 7:\n\n",t1as,"\n")
	println("\nTable 8:\n\n",t5as,"\n")
end

# output to tex
string2file(joinpath(outpath, "aucsummary.tex"), sra)
string2file(joinpath(outpath, "timesummary.tex"), smt)
string2file(joinpath(outpath, "aaucsummary.tex"), sraa)
string2file(joinpath(outpath, "testaucfull.tex"), tsas)
string2file(joinpath(outpath, "trainaucfull.tex"), tas)
string2file(joinpath(outpath, "top1aucfull.tex"), t1as)
string2file(joinpath(outpath, "top5aucfull.tex"), t5as)

### augmented ROC ###

# table 9 - mean scores of the second test with ranks
testaaurocdf = miss2hyphen!(rounddf(testaauc,2,2))
row = rounddf(valuedf[9,:],2,2)
rename!(row, :test, :dataset)
testaaurocdf = [testaaurocdf; row]
rename!(testaaurocdf, :IsoForest, :IForest)
testaaurocdf = rpaddf(testaaurocdf, 2)

ranktestaaurocdf = miss2hyphen!(rounddf(ranktestaauc,2,2))
tsaadf = mergedfs(testaaurocdf, ranktestaaurocdf)
tsaadf[:dataset] = shortnames
tsaas = df2tex(tsaadf,
	"Augmented AUC scores and ranks using the first hyperparameter selection criterion.",
	label = "tab:testaaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 10 - mean scores of the second test with ranks
trainaaurocdf = miss2hyphen!(rounddf(trainaauc,2,2))
row = rounddf(valuedf[10,:],2,2)
rename!(row, :test, :dataset)
trainaaurocdf = [trainaaurocdf; row]
rename!(trainaaurocdf, :IsoForest, :IForest)
trainaaurocdf = rpaddf(trainaaurocdf, 2)

ranktrainaaurocdf = miss2hyphen!(rounddf(ranktrainaauc,2,2))
taadf = mergedfs(trainaaurocdf, ranktrainaaurocdf)
taadf[:dataset] = shortnames
taas = df2tex(taadf,
	"Augmented AUC scores and ranks using the second hyperparameter selection criterion.",
	label = "tab:trainaaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 11 - mean scores of the third 1% test with ranks
top1aaurocdf = miss2hyphen!(rounddf(top1aauc,2,2))
row = rounddf(valuedf[9,:],2,2)
rename!(row, :test, :dataset)
top1aaurocdf = [top1aaurocdf; row]
rename!(top1aaurocdf, :IsoForest, :IForest)
top1aaurocdf = rpaddf(top1aaurocdf, 2)

ranktop1aaurocdf = miss2hyphen!(rounddf(ranktop1aauc,2,2))
t1aadf = mergedfs(top1aaurocdf, ranktop1aaurocdf)
t1aadf[:dataset] = shortnames
t1aas = df2tex(t1aadf,
	"Augmented AUC scores and ranks using the third hyperparameter selection criterion, using top 1\\% of samples.",
	label = "tab:top1aaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 12 - mean scores of the third 5% test with ranks
top5aaurocdf = miss2hyphen!(rounddf(top5aauc,2,2))
row = rounddf(valuedf[10,:],2,2)
rename!(row, :test, :dataset)
top5aaurocdf = [top5aaurocdf; row]
rename!(top5aaurocdf, :IsoForest, :IForest)
top5aaurocdf = rpaddf(top5aaurocdf, 2)

ranktop5aaurocdf = miss2hyphen!(rounddf(ranktop5aauc,2,2))
t5aadf = mergedfs(top5aaurocdf, ranktop5aaurocdf)
t5aadf[:dataset] = shortnames
t5aas = df2tex(t5aadf,
	"Augmented AUC scores and ranks using the third hyperparameter selection criterion, using top 5\\% of samples.",
	label = "tab:top5aaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

if verb
	# cl output
	println("\nTable 9:\n\n",tsaas,"\n")
	println("\nTable 10:\n\n",taas,"\n")
	println("\nTable 11:\n\n",t1aas,"\n")
	println("\nTable 12:\n\n",t5aas,"\n")
end

# output to tex
string2file(joinpath(outpath, "testaaucfull.tex"), tsaas)
string2file(joinpath(outpath, "trainaaucfull.tex"), taas)
string2file(joinpath(outpath, "top1aaucfull.tex"), t1aas)
string2file(joinpath(outpath, "top5aaucfull.tex"), t5aas)

### large time tables ###
# table 13 - mean fit times with ranks
fittdf = miss2hyphen!(rounddf(meanfitt,2,2))
row = rounddf(valuedf[5,:],2,2)
rename!(row, :test, :dataset)
fittdf = [fittdf; row]
rename!(fittdf, :IsoForest, :IForest)
fittdf = rpaddf(fittdf, 2)

rankfittdf = miss2hyphen!(rounddf(rankmeanfitt,2,2))
ftdf = mergedfs(fittdf, rankfittdf)
ftdf[:dataset] = shortnames
fts = df2tex(ftdf,
	"Mean fit times over datasets with ranks and average",
	label = "tab:ftfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 14 - mean scores of the third 5% test with ranks
predicttdf = miss2hyphen!(rounddf(meanpredictt,2,2))
row = rounddf(valuedf[6,:],2,2)
rename!(row, :test, :dataset)
predicttdf = [predicttdf; row]
rename!(predicttdf, :IsoForest, :IForest)
predicttdf = rpaddf(predicttdf, 2)

rankpredicttdf = miss2hyphen!(rounddf(rankmeanpredictt,2,2))
ptdf = mergedfs(predicttdf, rankpredicttdf)
ptdf[:dataset] = shortnames
pts = df2tex(ptdf,
	"Mean predict times over datasets with ranks and average",
	label = "tab:ptfull",
	fitcolumn = true, lasthline = true, firstvline = true)


if verb
	# cl output
	println("\nTable 13:\n\n",fts,"\n")
	println("\nTable 14:\n\n",pts,"\n")
end

# output to tex
string2file(joinpath(outpath, "ftfull.tex"), fts)
string2file(joinpath(outpath, "ptfull.tex"), pts)
