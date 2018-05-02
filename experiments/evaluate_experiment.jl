include("./eval.jl")

#data_path = "./data"
#outpath = "./output"
#evalpath = "./eval"
data_path = "/home/vit/vyzkum/anomaly_detection/data/aws/anomaly"
outpath = "/home/vit/vyzkum/anomaly_detection/data/aws/output"
evalpath = "/home/vit/vyzkum/anomaly_detection/data/aws/eval"
datasets = filter(s->s!="persistant-connection",readdir(data_path))
datasets = filter(s->s!="gisette", datasets)
mkpath(outpath)
mkpath(evalpath)

#experiment = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1:5

#algnames = ["kNN", "kNNPCA", "IsoForest", "AE", "VAE", "sVAE", "GAN", "fmGAN"]
algnames = ["kNN", "IsoForest", "AE", "VAE", "GAN", "fmGAN"]
#algnames = ["kNN", "kNNPCA"]

# create a dataframe with a row for each experiment
println("Computing basic experiment statistics...")
for dataset in datasets
	for alg in algnames
		fpath = joinpath(outpath, dataset)
		mkpath(fpath)
		f = joinpath(fpath, "$alg.csv")
		if !isfile(f)
			df = computedatasetstats(data_path, dataset, [alg])
			writetable(f, df)
			println("Computed and saved $(f)!")
		else
			println("$f is already present!")
		end
	end
end
println("Done.")

# first experiment
println("First experiment - rank algorithms on maximum auroc in an iteration, 
	average over iterations.")
println("")
maxauc = collectscores(outpath, algnames, maxauroc)
rankmaxauc = rankdf(maxauc)
writetable(joinpath(evalpath, "maxauc.csv"), maxauc);
writetable(joinpath(evalpath, "rankmaxauc.csv"), rankmaxauc);
showall(maxauc)
println("")
showall(rankmaxauc)
println("")
 
# second experiment
println("Second experiment - select hyperparameters on training dataset, then average
	the testing auroc for given hyperparameters over iterations.")
println("")
trainauc = collectscores(outpath, algnames, trainauroc)
ranktrainauc = rankdf(trainauc)
writetable(joinpath(evalpath, "trainauc.csv"), trainauc);
writetable(joinpath(evalpath, "ranktrainauc.csv"), ranktrainauc);
showall(trainauc)
println("")
showall(ranktrainauc)
println("")

# third experiment
println("Third experiment - select hyperparameters on training dataset based on top 5% precision,
	then average the testing auroc for given hyperparameters over iterations.")
println("")
top5auc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_5p))
ranktop5auc = rankdf(top5auc)
writetable(joinpath(evalpath, "top5auc.csv"), top5auc);
writetable(joinpath(evalpath, "ranktop5auc.csv"), ranktop5auc);
showall(top5auc)
println("")
showall(ranktop5auc)
println("")

# 4th
println("Fourth experiment - select hyperparameters on training dataset based on top 1% precision,
	then average the testing auroc for given hyperparameters over iterations.")
println("")
top1auc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_1p))
ranktop1auc = rankdf(top1auc)
writetable(joinpath(evalpath, "top1auc.csv"), top1auc);
writetable(joinpath(evalpath, "ranktop1auc.csv"), ranktop1auc);
showall(top1auc)
println("")
showall(ranktop1auc)
println("")

# fifth experiment
println("Fifth experiment - rank by mean fit time.")
println("")
meanfitt = collectscores(outpath, algnames, (x,y)->meantime(x,y,"fit_time"))
showall(meanfitt)
println("")
rankmeanfitt = rankdf(meanfitt, false)
writetable(joinpath(evalpath, "meanfitt.csv"), meanfitt);
writetable(joinpath(evalpath, "rankmeanfitt.csv"), rankmeanfitt);
showall(rankmeanfitt)
println("")

# sixth experiment
println("Sixth experiment - rank by mean predict time.")
println("")
meanpredictt = collectscores(outpath, algnames, (x,y)->meantime(x,y,"predict_time"))
rankmeanpredictt = rankdf(meanpredictt, false)
writetable(joinpath(evalpath, "meanpredictt.csv"), meanpredictt);
writetable(joinpath(evalpath, "rankmeanpredictt.csv"), rankmeanpredictt);
showall(meanpredictt)
println("")
showall(rankmeanpredictt)
println("")

# 7th experiment
println("7th experiment - rank algorithms on maximum augmented auroc in an iteration, 
	average over iterations.")
println("")
maxaauc = collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,"augmented"))
rankmaxaauc = rankdf(maxaauc)
writetable(joinpath(evalpath, "maxaauc.csv"), maxaauc);
writetable(joinpath(evalpath, "rankmaxaauc.csv"), rankmaxaauc);
showall(maxaauc)
println("")
showall(rankmaxaauc)
println("")

# 8th experiment
println("8th experiment - select hyperparameters on training dataset, then average
	the testing augmented auroc for given hyperparameters over iterations.")
println("")
trainaauc = collectscores(outpath, algnames, (x,y) -> trainauroc(x,y,"augmented"))
ranktrainaauc = rankdf(trainaauc)
writetable(joinpath(evalpath, "trainaauc.csv"), trainaauc);
writetable(joinpath(evalpath, "ranktrainaauc.csv"), ranktrainaauc);
showall(trainaauc)
println("")
showall(ranktrainaauc)
println("")

# 9th experiment
println("9th experiment - select hyperparameters on training dataset based on top 5% precision,
	then average the testing augmented auroc for given hyperparameters over iterations.")
println("")
top5aauc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_5p,"augmented"))
ranktop5aauc = rankdf(top5aauc)
writetable(joinpath(evalpath, "top5aauc.csv"), top5aauc);
writetable(joinpath(evalpath, "ranktop5aauc.csv"), ranktop5aauc);
showall(top5aauc)
println("")
showall(ranktop5aauc)
println("")

# 10th
println("10th experiment - select hyperparameters on training dataset based on top 1% precision,
	then average the testing augmented auroc for given hyperparameters over iterations.")
println("")
top1aauc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_1p,"augmented"))
ranktop1aauc = rankdf(top1aauc)
writetable(joinpath(evalpath, "top1aauc.csv"), top1aauc);
writetable(joinpath(evalpath, "ranktop1aauc.csv"), ranktop1aauc);
showall(top1aauc)
println("")
showall(ranktop1aauc)
println("")

# summary of all experiments
valuedf = createdf(algnames)
rename!(valuedf, :dataset, :test)
push!(valuedf, cat(1,["test auc"],[x[1] for x in colwise(missmean, maxauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["train auc"],[x[1] for x in colwise(missmean, trainauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 1%"],[x[1] for x in colwise(missmean, top1auc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 5%"],[x[1] for x in colwise(missmean, top5auc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["mean fit time"],[x[1] for x in colwise(missmean, meanfitt[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["mean predict time"],[x[1] for x in colwise(missmean, meanpredictt[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["test auc - augmented"],[x[1] for x in colwise(missmean, maxaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["train auc - augmented"],[x[1] for x in colwise(missmean, trainaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 1% - augmented"],[x[1] for x in colwise(missmean, top1aauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 5% - augmented"],[x[1] for x in colwise(missmean, top5aauc[[Symbol(alg) for alg in algnames]])]))
writetable(joinpath(evalpath, "valuesummary.csv"), valuedf);

rankeddf = createdf(algnames)
rankeddf = [rankeddf; rankmaxauc[end,:]]
rankeddf[:dataset][end] = "test auc"
rankeddf = [rankeddf; ranktrainauc[end,:]]
rankeddf[:dataset][end] = "train auc"
rankeddf = [rankeddf; ranktop1auc[end,:]]
rankeddf[:dataset][end] = "top 1%"
rankeddf = [rankeddf; ranktop5auc[end,:]]
rankeddf[:dataset][end] = "top 5%"
rankeddf = [rankeddf; rankmeanfitt[end,:]]
rankeddf[:dataset][end] = "mean fit time"
rankeddf = [rankeddf; rankmeanpredictt[end,:]]
rankeddf[:dataset][end] = "mean predict time"
rankeddf = [rankeddf; rankmaxaauc[end,:]]
rankeddf[:dataset][end] = "test auc - augmented"
rankeddf = [rankeddf; ranktrainaauc[end,:]]
rankeddf[:dataset][end] = "train auc - augmented"
rankeddf = [rankeddf; ranktop1aauc[end,:]]
rankeddf[:dataset][end] = "top 1% - augmented"
rankeddf = [rankeddf; ranktop5aauc[end,:]]
rankeddf[:dataset][end] = "top 5% - augmented"
rename!(rankeddf, :dataset, :test)
writetable(joinpath(evalpath, "ranksummary.csv"), rankeddf);

println("")
println("Summary of mean test values:")
showall(rounddf(valuedf, 2, 2))
println("")
println("Summary of mean ranks:")
showall(rounddf(rankeddf, 2, 2))
println("")