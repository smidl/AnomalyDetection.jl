(length(ARGS) > 0)? ((ARGS[1] == "v")? verb=true:verb=false ) : verb = false

ARGS = ["d"]
include("evaluate_experiment.jl")

# first experiment
maxauc = collectscores(outpath, algnames, maxauroc)
rankmaxauc = rankdf(maxauc)
writetable(joinpath(evalpath, "maxauc.csv"), maxauc);
writetable(joinpath(evalpath, "rankmaxauc.csv"), rankmaxauc);
if verb
	println("First experiment - rank algorithms on maximum auroc in an iteration,
	average over iterations.")
	println("")
	showall(maxauc)
	println("")
	showall(rankmaxauc)
	println("")
end

# 1.5th experiment - select hyperparameters by max mean auroc over testing data
testauc = collectscores(outpath, algnames, (x,y) -> testauroc(x,y,"normal","test"))
ranktestauc = rankdf(testauc)
writetable(joinpath(evalpath, "testauc.csv"), testauc);
writetable(joinpath(evalpath, "ranktestauc.csv"), ranktestauc);
if verb
	println("1.5th experiment - select hyperparameters on testing dataset, then average
	the testing auroc for given hyperparameters over iterations.")
	println("")
	showall(testauc)
	println("")
	showall(ranktestauc)
	println("")
end

# second experiment
trainauc = collectscores(outpath, algnames, testauroc)
ranktrainauc = rankdf(trainauc)
writetable(joinpath(evalpath, "trainauc.csv"), trainauc);
writetable(joinpath(evalpath, "ranktrainauc.csv"), ranktrainauc);
if verb
	println("Second experiment - select hyperparameters on training dataset, then average
	the testing auroc for given hyperparameters over iterations.")
	println("")
	showall(trainauc)
	println("")
	showall(ranktrainauc)
	println("")
end

# third experiment
top5auc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_5p))
ranktop5auc = rankdf(top5auc)
writetable(joinpath(evalpath, "top5auc.csv"), top5auc);
writetable(joinpath(evalpath, "ranktop5auc.csv"), ranktop5auc);
if verb
	println("Third experiment - select hyperparameters on training dataset based on top 5% precision,
	then average the testing auroc for given hyperparameters over iterations.")
	println("")
	showall(top5auc)
	println("")
	showall(ranktop5auc)
	println("")
end

# 4th
top1auc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_1p))
ranktop1auc = rankdf(top1auc)
writetable(joinpath(evalpath, "top1auc.csv"), top1auc);
writetable(joinpath(evalpath, "ranktop1auc.csv"), ranktop1auc);
if verb
	println("Fourth experiment - select hyperparameters on training dataset based on top 1% precision,
		then average the testing auroc for given hyperparameters over iterations.")
	println("")
	showall(top1auc)
	println("")
	showall(ranktop1auc)
	println("")
end

# fifth experiment
meanfitt = collectscores(outpath, algnames, (x,y)->meantime(x,y,"fit_time"))
rankmeanfitt = rankdf(meanfitt, false)
writetable(joinpath(evalpath, "meanfitt.csv"), meanfitt);
writetable(joinpath(evalpath, "rankmeanfitt.csv"), rankmeanfitt);
if verb
	println("Fifth experiment - rank by mean fit time.")
	println("")
	showall(meanfitt)
	println("")
	showall(rankmeanfitt)
	println("")
end

# sixth experiment
meanpredictt = collectscores(outpath, algnames, (x,y)->meantime(x,y,"predict_time"))
rankmeanpredictt = rankdf(meanpredictt, false)
writetable(joinpath(evalpath, "meanpredictt.csv"), meanpredictt);
writetable(joinpath(evalpath, "rankmeanpredictt.csv"), rankmeanpredictt);
if verb
	println("Sixth experiment - rank by mean predict time.")
	println("")
	showall(meanpredictt)
	println("")
	showall(rankmeanpredictt)
	println("")
end

# 7th experiment
maxaauc = collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,"augmented"))
rankmaxaauc = rankdf(maxaauc)
writetable(joinpath(evalpath, "maxaauc.csv"), maxaauc);
writetable(joinpath(evalpath, "rankmaxaauc.csv"), rankmaxaauc);
if verb
	println("7th experiment - rank algorithms on maximum augmented auroc in an iteration,
		average over iterations.")
	println("")
	showall(maxaauc)
	println("")
	showall(rankmaxaauc)
	println("")
end

# 7.5th experiment - select hyperparameters by max mean auroc over testing data
testaauc = collectscores(outpath, algnames, (x,y) -> testauroc(x,y,"augmented","test"))
ranktestaauc = rankdf(testaauc)
writetable(joinpath(evalpath, "testaauc.csv"), testaauc);
writetable(joinpath(evalpath, "ranktestaauc.csv"), ranktestaauc);
if verb
	println("7.5th experiment - select hyperparameters on testing dataset, then average
		the testing aauroc for given hyperparameters over iterations.")
	println("")
	showall(testaauc)
	println("")
	showall(ranktestaauc)
	println("")
end

# 8th experiment
trainaauc = collectscores(outpath, algnames, (x,y) -> testauroc(x,y,"augmented"))
ranktrainaauc = rankdf(trainaauc)
writetable(joinpath(evalpath, "trainaauc.csv"), trainaauc);
writetable(joinpath(evalpath, "ranktrainaauc.csv"), ranktrainaauc);
if verb
	println("8th experiment - select hyperparameters on training dataset, then average
		the testing augmented auroc for given hyperparameters over iterations.")
	println("")
	showall(trainaauc)
	println("")
	showall(ranktrainaauc)
	println("")
end

# 9th experiment
top5aauc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_5p,"augmented"))
ranktop5aauc = rankdf(top5aauc)
writetable(joinpath(evalpath, "top5aauc.csv"), top5aauc);
writetable(joinpath(evalpath, "ranktop5aauc.csv"), ranktop5aauc);
if verb
	println("9th experiment - select hyperparameters on training dataset based on top 5% precision,
		then average the testing augmented auroc for given hyperparameters over iterations.")
	println("")
	showall(top5aauc)
	println("")
	showall(ranktop5aauc)
	println("")
end

# 10th
top1aauc = collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_1p,"augmented"))
ranktop1aauc = rankdf(top1aauc)
writetable(joinpath(evalpath, "top1aauc.csv"), top1aauc);
writetable(joinpath(evalpath, "ranktop1aauc.csv"), ranktop1aauc);
if verb
	println("10th experiment - select hyperparameters on training dataset based on top 1% precision,
		then average the testing augmented auroc for given hyperparameters over iterations.")
	println("")
	showall(top1aauc)
	println("")
	showall(ranktop1aauc)
	println("")
end

# summary of all experiments
valuedf = createdf(algnames)
rename!(valuedf, :dataset, :test)
push!(valuedf, cat(1,["max auc"],[x[1] for x in colwise(missmean, maxauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["test auc"],[x[1] for x in colwise(missmean, testauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["train auc"],[x[1] for x in colwise(missmean, trainauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 1%"],[x[1] for x in colwise(missmean, top1auc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 5%"],[x[1] for x in colwise(missmean, top5auc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["mean fit time"],[x[1] for x in colwise(missmean, meanfitt[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["mean predict time"],[x[1] for x in colwise(missmean, meanpredictt[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["max auc - augmented"],[x[1] for x in colwise(missmean, maxaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["test auc - augmented"],[x[1] for x in colwise(missmean, testaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["train auc - augmented"],[x[1] for x in colwise(missmean, trainaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 1% - augmented"],[x[1] for x in colwise(missmean, top1aauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 5% - augmented"],[x[1] for x in colwise(missmean, top5aauc[[Symbol(alg) for alg in algnames]])]))
writetable(joinpath(evalpath, "valuesummary.csv"), valuedf);

rankeddf = createdf(algnames)
rankeddf = [rankeddf; rankmaxauc[end,:]]
rankeddf[:dataset][end] = "max auc"
rankeddf = [rankeddf; ranktestauc[end,:]]
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
rankeddf[:dataset][end] = "max auc - augmented"
rankeddf = [rankeddf; ranktestaauc[end,:]]
rankeddf[:dataset][end] = "test auc - augmented"
rankeddf = [rankeddf; ranktrainaauc[end,:]]
rankeddf[:dataset][end] = "train auc - augmented"
rankeddf = [rankeddf; ranktop1aauc[end,:]]
rankeddf[:dataset][end] = "top 1% - augmented"
rankeddf = [rankeddf; ranktop5aauc[end,:]]
rankeddf[:dataset][end] = "top 5% - augmented"
rename!(rankeddf, :dataset, :test)
writetable(joinpath(evalpath, "ranksummary.csv"), rankeddf);

if verb
	println("")
	println("Summary of mean test values:")
	showall(rounddf(valuedf, 2, 2))
	println("")
	println("Summary of mean ranks:")
	showall(rounddf(rankeddf, 2, 2))
	println("")
end