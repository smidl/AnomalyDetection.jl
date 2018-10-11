# run as 'julia create_tex_overview.jl path'
# this will create a directory with a tex and pdf file with tables and graphs 
# for results stored in path/eval
include("eval.jl")
#cdalgnames = ["kNN", "IForest", "AE", "VAE", "GAN", "fmGAN", "VAEensemble"]
cdalgnames = ["kNN", "IForest", "AE", "VAE", "GAN", "fmGAN"]
path = ARGS[1]
evalpath = joinpath(path, "eval")
texpath = joinpath(path, "texout")
mkpath(texpath)

# load all the files in evalpath
testauc = loadtable(joinpath(evalpath, "testauc.csv"), 2)
ranktestauc = loadtable(joinpath(evalpath, "ranktestauc.csv"), 2)
trainauc = loadtable(joinpath(evalpath, "trainauc.csv"), 2);
ranktrainauc = loadtable(joinpath(evalpath, "ranktrainauc.csv"), 2);
top5auc = loadtable(joinpath(evalpath, "top5auc.csv"), 2);
ranktop5auc = loadtable(joinpath(evalpath, "ranktop5auc.csv"), 2);
top1auc = loadtable(joinpath(evalpath, "top1auc.csv"), 2);
ranktop1auc = loadtable(joinpath(evalpath, "ranktop1auc.csv"), 2);
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
top1aauc = loadtable(joinpath(evalpath, "top1aauc.csv"), 2);
ranktop1aauc = loadtable(joinpath(evalpath, "ranktop1aauc.csv"), 2);
valuedf = loadtable(joinpath(evalpath, "valuesummary.csv"), 2);
rankeddf = loadtable(joinpath(evalpath, "ranksummary.csv"), 2);


#const shortnames = ["aba", "blo", "brc", "brt", "car", "eco", "gla", "hab", "ion", "iri", 
#	"iso", "let", "lib", "mad", "mag", "min", "mul", "mus", "pag", "par", "pen", "pim", "son", 
#	"spe", "ssa", "sse", "ssh", "sve", "syn", "ver", "wal", "wa1", "wa2", "win", "yea", "avg"]

function shortnames!(df, nametable)
	for (i, dataset) in enumerate(df[:dataset])
		shortname = dataset[1:min(3,length(dataset))]
		try
			shortname = nametable[dataset]
		end
		df[:dataset][i] = shortname
	end
end

const nametable = Dict(
	"abalone" => "aba",
	"blood-transfusion" => "blo",
	"breast-cancer-wisconsin" => "brc",
	"breast-tissue" => "brt",
	"cardiotocography" => "car",
	"ecoli" => "eco",
	"gisette" => "gis",
	"glass" => "gla",
	"haberman" => "hab",
	"ionosphere" => "ion",
	"iris" => "iri",
	"isolet" => "iso",
	"letter-recognition" => "let",
	"libras" => "lib",
	"madelon" => "mad",
	"magic-telescope" => "mag",
	"miniboone" => "min",
	"multiple-features" => "mul",
	"musk-2" => "mus",
	"page-blocks" => "pag",
	"parkinsons" => "par",
	"pendigits" => "pen",
	"pima-indians" => "pim",
	"sonar" => "son",
	"spect-heart" => "spe",
	"statlog-satimage" => "ssa",
	"statlog-segment" => "sse",
	"statlog-shuttle" => "ssh",
	"statlog-vehicle" => "sve",
	"synthetic-control-chart" => "syn",
	"vertebral-column" => "ver",
	"wall-following-robot" => "wal",
	"waveform-1" => "wa1",
	"waveform-2" => "wa2",
	"wine" => "win",
	"yeast" => "yea",
	"average" => "avg"
	)

#(size(ARGS,1)>1)? v = ARGS[1] : v = "verb"

# table 2 - mean ranks on normal auroc
rankaurocsum = rounddf(rankeddf[1:4,:], 2, 2)
rankaurocsum[:test][3] = "top 5\\%"
rankaurocsum[:test][4] = "top 1\\%"
rename!(rankaurocsum, :IsoForest => :IForest)
rename!(rankaurocsum, :test => Symbol(" "))
rankaurocsum = rpaddf(rankaurocsum, 2)
sra = df2tex(rankaurocsum, "Average ranks of algorithms for different hyperparameter selection criteria.",
	label = "tab:aucsummary",
	firstvline = true)

# table 3 - mean fit/predict times
meantimessum = rounddf(valuedf[5:6,:],2,2)
meantimessum[:test][1] = "\$t_f\$ [s]"
meantimessum[:test][2] = "\$t_p\$ [s]"
rename!(meantimessum, :IsoForest => :IForest)
rename!(meantimessum, :test => Symbol(" "))
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
rename!(rankaaurocsum, :IsoForest => :IForest)
rename!(rankaaurocsum, :test => Symbol(" "))
rankaaurocsum = rpaddf(rankaaurocsum, 2)
sraa = df2tex(rankaaurocsum, 
	"Average ranks of algorithms for different hyperparameter selection criteria using the augmented AUROC.",
	label = "tab:aaucsummary",
	firstvline = true)

# table 5 - mean scores of the first test with ranks
testaurocdf = miss2hyphen!(rounddf(testauc,2,2))
row = rounddf(valuedf[1,:],2,2)
rename!(row, :test => :dataset)
testaurocdf = [testaurocdf; row]
rename!(testaurocdf, :IsoForest => :IForest)
testaurocdf = rpaddf(testaurocdf, 2)

ranktestaurocdf = miss2hyphen!(rounddf(ranktestauc,2,2))
tsadf = mergedfs(testaurocdf, ranktestaurocdf)
tsadf[:dataset][end] = "average"
shortnames!(tsadf, nametable)
tsas = df2tex(tsadf,
	"AUROC scores and ranks of algorithms using the first hyperparameter selection criterion. The last line is an average.",
	label = "tab:testaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 6 - mean scores of the second test with ranks
trainaurocdf = miss2hyphen!(rounddf(trainauc,2,2))
row = rounddf(valuedf[2,:],2,2)
rename!(row, :test => :dataset)
trainaurocdf = [trainaurocdf; row]
rename!(trainaurocdf, :IsoForest => :IForest)
trainaurocdf = rpaddf(trainaurocdf, 2)

ranktrainaurocdf = miss2hyphen!(rounddf(ranktrainauc,2,2))
tadf = mergedfs(trainaurocdf, ranktrainaurocdf)
tadf[:dataset][end] = "average"
shortnames!(tadf, nametable)
tas = df2tex(tadf,
	"AUROC scores and ranks of algorithms using the second hyperparameter selection criterion.",
	label = "tab:trainaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 7 - mean scores of the third 1% test with ranks
top1aurocdf = miss2hyphen!(rounddf(top1auc,2,2))
row = rounddf(valuedf[3,:],2,2)
rename!(row, :test => :dataset)
top1aurocdf = [top1aurocdf; row]
rename!(top1aurocdf, :IsoForest => :IForest)
top1aurocdf = rpaddf(top1aurocdf, 2)

ranktop1aurocdf = miss2hyphen!(rounddf(ranktop1auc,2,2))
t1adf = mergedfs(top1aurocdf, ranktop1aurocdf)
t1adf[:dataset][end] = "average"
shortnames!(t1adf, nametable)
t1as = df2tex(t1adf,
	"AUROC scores and ranks of algorithms using the third hyperparameter selection criterion, using top 1\\% of samples.",
	label = "tab:top1aucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 8 - mean scores of the third 5% test with ranks
top5aurocdf = miss2hyphen!(rounddf(top5auc,2,2))
row = rounddf(valuedf[4,:],2,2)
rename!(row, :test => :dataset)
top5aurocdf = [top5aurocdf; row]
rename!(top5aurocdf, :IsoForest => :IForest)
top5aurocdf = rpaddf(top5aurocdf, 2)

ranktop5aurocdf = miss2hyphen!(rounddf(ranktop5auc,2,2))
t5adf = mergedfs(top5aurocdf, ranktop5aurocdf)
t5adf[:dataset][end] = "average"
shortnames!(t5adf, nametable)
t5as = df2tex(t5adf,
	"AUROC scores and ranks of algorithms using the third hyperparameter selection criterion, using top 5\\% of samples.",
	label = "tab:top5aucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# output to tex
string2file(joinpath(texpath, "aucsummary.tex"), sra)
string2file(joinpath(texpath, "timesummary.tex"), smt)
string2file(joinpath(texpath, "aaucsummary.tex"), sraa)
string2file(joinpath(texpath, "testaucfull.tex"), tsas)
string2file(joinpath(texpath, "trainaucfull.tex"), tas)
string2file(joinpath(texpath, "top1aucfull.tex"), t1as)
string2file(joinpath(texpath, "top5aucfull.tex"), t5as)

### augmented ROC ###

# table 9 - mean scores of the second test with ranks
testaaurocdf = miss2hyphen!(rounddf(testaauc,2,2))
row = rounddf(valuedf[9,:],2,2)
rename!(row, :test => :dataset)
testaaurocdf = [testaaurocdf; row]
rename!(testaaurocdf, :IsoForest => :IForest)
testaaurocdf = rpaddf(testaaurocdf, 2)

ranktestaaurocdf = miss2hyphen!(rounddf(ranktestaauc,2,2))
tsaadf = mergedfs(testaaurocdf, ranktestaaurocdf)
tsaadf[:dataset][end] = "average"
shortnames!(tsaadf, nametable)
tsaas = df2tex(tsaadf,
	"Augmented AUC scores and ranks using the first hyperparameter selection criterion.",
	label = "tab:testaaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 10 - mean scores of the second test with ranks
trainaaurocdf = miss2hyphen!(rounddf(trainaauc,2,2))
row = rounddf(valuedf[10,:],2,2)
rename!(row, :test => :dataset)
trainaaurocdf = [trainaaurocdf; row]
rename!(trainaaurocdf, :IsoForest => :IForest)
trainaaurocdf = rpaddf(trainaaurocdf, 2)

ranktrainaaurocdf = miss2hyphen!(rounddf(ranktrainaauc,2,2))
taadf = mergedfs(trainaaurocdf, ranktrainaaurocdf)
taadf[:dataset][end] = "average"
shortnames!(taadf, nametable)
taas = df2tex(taadf,
	"Augmented AUC scores and ranks using the second hyperparameter selection criterion.",
	label = "tab:trainaaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 11 - mean scores of the third 1% test with ranks
top1aaurocdf = miss2hyphen!(rounddf(top1aauc,2,2))
row = rounddf(valuedf[9,:],2,2)
rename!(row, :test => :dataset)
top1aaurocdf = [top1aaurocdf; row]
rename!(top1aaurocdf, :IsoForest => :IForest)
top1aaurocdf = rpaddf(top1aaurocdf, 2)

ranktop1aaurocdf = miss2hyphen!(rounddf(ranktop1aauc,2,2))
t1aadf = mergedfs(top1aaurocdf, ranktop1aaurocdf)
t1aadf[:dataset][end] = "average"
shortnames!(t1aadf, nametable)
t1aas = df2tex(t1aadf,
	"Augmented AUC scores and ranks using the third hyperparameter selection criterion, using top 1\\% of samples.",
	label = "tab:top1aaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 12 - mean scores of the third 5% test with ranks
top5aaurocdf = miss2hyphen!(rounddf(top5aauc,2,2))
row = rounddf(valuedf[10,:],2,2)
rename!(row, :test => :dataset)
top5aaurocdf = [top5aaurocdf; row]
rename!(top5aaurocdf, :IsoForest => :IForest)
top5aaurocdf = rpaddf(top5aaurocdf, 2)

ranktop5aaurocdf = miss2hyphen!(rounddf(ranktop5aauc,2,2))
t5aadf = mergedfs(top5aaurocdf, ranktop5aaurocdf)
t5aadf[:dataset][end] = "average"
shortnames!(t5aadf, nametable)
t5aas = df2tex(t5aadf,
	"Augmented AUC scores and ranks using the third hyperparameter selection criterion, using top 5\\% of samples.",
	label = "tab:top5aaucfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# output to tex
string2file(joinpath(texpath, "testaaucfull.tex"), tsaas)
string2file(joinpath(texpath, "trainaaucfull.tex"), taas)
string2file(joinpath(texpath, "top1aaucfull.tex"), t1aas)
string2file(joinpath(texpath, "top5aaucfull.tex"), t5aas)

### large time tables ###
# table 13 - mean fit times with ranks
fittdf = miss2hyphen!(rounddf(meanfitt,2,2))
row = rounddf(valuedf[5,:],2,2)
rename!(row, :test => :dataset)
fittdf = [fittdf; row]
rename!(fittdf, :IsoForest => :IForest)
fittdf = rpaddf(fittdf, 2)

rankfittdf = miss2hyphen!(rounddf(rankmeanfitt,2,2))
ftdf = mergedfs(fittdf, rankfittdf)
ftdf[:dataset][end] = "average"
shortnames!(ftdf, nametable)
fts = df2tex(ftdf,
	"Mean fit times over datasets with ranks and average",
	label = "tab:ftfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# table 14 - mean scores of the third 5% test with ranks
predicttdf = miss2hyphen!(rounddf(meanpredictt,2,2))
row = rounddf(valuedf[6,:],2,2)
rename!(row, :test => :dataset)
predicttdf = [predicttdf; row]
rename!(predicttdf, :IsoForest => :IForest)
predicttdf = rpaddf(predicttdf, 2)

rankpredicttdf = miss2hyphen!(rounddf(rankmeanpredictt,2,2))
ptdf = mergedfs(predicttdf, rankpredicttdf)
ptdf[:dataset][end] = "average"
shortnames!(ptdf, nametable)
pts = df2tex(ptdf,
	"Mean predict times over datasets with ranks and average",
	label = "tab:ptfull",
	fitcolumn = true, lasthline = true, firstvline = true)

# output to tex
string2file(joinpath(texpath, "ftfull.tex"), fts)
string2file(joinpath(texpath, "ptfull.tex"), pts)

### CRITICAL DIFFERENCE DIAGRAMS ###

#println(rankaurocsum[1,2:end])
#println(typeof(rankaurocsum[1,2:end]))
#println(typeof(Float64.(rankaurocsum[1,2:end])))

#cv = 1.1578 # Nemenyi 0.1 critical value
cv = 1.2746 # Nemenyi 0.05 critical value
tacds = ranks2tikzcd(convert(Array, rankaurocsum[1,2:end]), cdalgnames, cv, "Critical difference diagram for the first hyperparameter selection criterion.", label = "fig:cdd1")
tracds = ranks2tikzcd(convert(Array, rankaurocsum[2,2:end]), cdalgnames, cv, "Critical difference diagram for the second hyperparameter selection criterion.", label = "fig:cdd2")
t5acds = ranks2tikzcd(convert(Array, rankaurocsum[3,2:end]), cdalgnames, cv, "Critical difference diagram for the third hyperparameter selection criterion at 5\\% most anomalous samples.", label = "fig:cdd3")
t1acds = ranks2tikzcd(convert(Array, rankaurocsum[4,2:end]), cdalgnames, cv, "Critical difference diagram for the third hyperparameter selection criterion at 1\\% most anomalous samples.", label = "fig:cdd4")

# augmented cds
taacds = ranks2tikzcd(convert(Array, rankaaurocsum[1,2:end]), cdalgnames, cv, "Critical difference diagram for the first hyperparameter selection criterion.", label = "fig:cdd5")
traacds = ranks2tikzcd(convert(Array, rankaaurocsum[2,2:end]), cdalgnames, cv, "Critical difference diagram for the second hyperparameter selection criterion.", label = "fig:cdd6")
t5aacds = ranks2tikzcd(convert(Array, rankaaurocsum[3,2:end]), cdalgnames, cv, "Critical difference diagram for the third hyperparameter selection criterion at 5\\% most anomalous samples.", label = "fig:cdd7")
t1aacds = ranks2tikzcd(convert(Array, rankaaurocsum[4,2:end]), cdalgnames, cv, "Critical difference diagram for the third hyperparameter selection criterion at 1\\% most anomalous samples.", label = "fig:cdd8")

# output to txt
string2file(joinpath(texpath, "cdd1.tex"), tacds)
string2file(joinpath(texpath, "cdd2.tex"), tracds)
string2file(joinpath(texpath, "cdd3.tex"), t5acds)
string2file(joinpath(texpath, "cdd4.tex"), t1acds)

string2file(joinpath(texpath, "cdd5.tex"), taacds)
string2file(joinpath(texpath, "cdd6.tex"), traacds)
string2file(joinpath(texpath, "cdd7.tex"), t5aacds)
string2file(joinpath(texpath, "cdd8.tex"), t1aacds)

### TEX and PDF output ###
texs = "\\documentclass{article}
\\usepackage{tikz}

\\begin{document}
	\\input{aucsummary.tex}
	\\input{timesummary.tex}
	\\input{aaucsummary.tex}

	\\input{cdd1.tex}
	\\input{cdd2.tex}
	\\input{cdd3.tex}
	\\input{cdd4.tex}

	\\input{testaucfull.tex}
	\\input{trainaucfull.tex}
	\\input{top5aucfull.tex}
	\\input{top1aucfull.tex}

	\\input{cdd5.tex}
	\\input{cdd6.tex}
	\\input{cdd7.tex}
	\\input{cdd8.tex}

	\\input{testaaucfull.tex}
	\\input{trainaaucfull.tex}
	\\input{top5aaucfull.tex}
	\\input{top1aaucfull.tex}

	\\input{ftfull.tex}
	\\input{ptfull.tex}
\\end{document}
"
string2file(joinpath(texpath, "overview.tex"), texs)

# finally, compile with pdflatex
cd(texpath)
run(`pdflatex overview.tex`)