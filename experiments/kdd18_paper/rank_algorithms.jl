# run as 'julia rank_algorithms path [v]'
# create ranking tables
(length(ARGS) > 1)? ((ARGS[2] == "v")? verb=true:verb=false ) : verb = false

ARGS = [ARGS[1], "d"]
include("evaluate_experiment.jl")

# 1st experiment - select hyperparameters by max mean auroc over testing data
#testauc  = rounddf(collectscores(outpath, algnames, (x,y) -> testauroc(x,y,"normal","test")),2,2)
testauc  = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:test_auroc,:test_auroc)),2,2)
ranktestauc = rankdf(testauc)
CSV.write(joinpath(evalpath, "testauc.csv"), testauc);
CSV.write(joinpath(evalpath, "ranktestauc.csv"), ranktestauc);
if verb
	println("1st experiment - select hyperparameters on testing dataset, then average
	the testing auroc for given hyperparameters over iterations.")
	println("")
	showall(testauc)
	println("")
	showall(ranktestauc)
	println("")
end

# second experiment
#trainauc = rounddf(collectscores(outpath, algnames, maxauroc),2,2)
trainauc = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:train_auroc,:test_auroc)),2,2)
ranktrainauc = rankdf(trainauc)
CSV.write(joinpath(evalpath, "trainauc.csv"), trainauc);
CSV.write(joinpath(evalpath, "ranktrainauc.csv"), ranktrainauc);
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
#top5auc = rounddf(collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_5p)),2,2)
top5auc = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:top_5p,:test_auroc)),2,2)
ranktop5auc = rankdf(top5auc)
CSV.write(joinpath(evalpath, "top5auc.csv"), top5auc);
CSV.write(joinpath(evalpath, "ranktop5auc.csv"), ranktop5auc);
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
#top1auc = rounddf(collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_1p)),2,2)
top1auc = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:top_1p,:test_auroc)),2,2)
ranktop1auc = rankdf(top1auc)
CSV.write(joinpath(evalpath, "top1auc.csv"), top1auc);
CSV.write(joinpath(evalpath, "ranktop1auc.csv"), ranktop1auc);
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
meanfitt = rounddf(collectscores(outpath, algnames, (x,y)->meantime(x,y,"fit_time")),2,2)
rankmeanfitt = rankdf(meanfitt, false)
CSV.write(joinpath(evalpath, "meanfitt.csv"), meanfitt);
CSV.write(joinpath(evalpath, "rankmeanfitt.csv"), rankmeanfitt);
if verb
	println("Fifth experiment - rank by mean fit time.")
	println("")
	showall(meanfitt)
	println("")
	showall(rankmeanfitt)
	println("")
end

# sixth experiment
meanpredictt = rounddf(collectscores(outpath, algnames, (x,y)->meantime(x,y,"predict_time")),2,2)
rankmeanpredictt = rankdf(meanpredictt, false)
CSV.write(joinpath(evalpath, "meanpredictt.csv"), meanpredictt);
CSV.write(joinpath(evalpath, "rankmeanpredictt.csv"), rankmeanpredictt);
if verb
	println("Sixth experiment - rank by mean predict time.")
	println("")
	showall(meanpredictt)
	println("")
	showall(rankmeanpredictt)
	println("")
end

# 7th experiment - select hyperparameters by max mean auroc over testing data
#testaauc = rounddf(collectscores(outpath, algnames, (x,y) -> testauroc(x,y,"augmented","test")),2,2)
testaauc  = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:test_aauroc,:test_aauroc)),2,2)
ranktestaauc = rankdf(testaauc)
CSV.write(joinpath(evalpath, "testaauc.csv"), testaauc);
CSV.write(joinpath(evalpath, "ranktestaauc.csv"), ranktestaauc);
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
#trainaauc = rounddf(collectscores(outpath, algnames, (x,y) -> testauroc(x,y,"augmented")),2,2)
trainaauc  = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:train_aauroc,:test_aauroc)),2,2)
ranktrainaauc = rankdf(trainaauc)
CSV.write(joinpath(evalpath, "trainaauc.csv"), trainaauc);
CSV.write(joinpath(evalpath, "ranktrainaauc.csv"), ranktrainaauc);
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
#top5aauc = rounddf(collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_5p,"augmented")),2,2)
top5aauc  = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:top_5p,:test_aauroc)),2,2)
ranktop5aauc = rankdf(top5aauc)
CSV.write(joinpath(evalpath, "top5aauc.csv"), top5aauc);
CSV.write(joinpath(evalpath, "ranktop5aauc.csv"), ranktop5aauc);
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
#top1aauc = rounddf(collectscores(outpath, algnames,(x,y)-> topprec(x,y,:top_1p,"augmented")),2,2)
top1aauc  = rounddf(collectscores(outpath, algnames, (x,y) -> maxauroc(x,y,:top_1p,:test_aauroc)),2,2)
ranktop1aauc = rankdf(top1aauc)
CSV.write(joinpath(evalpath, "top1aauc.csv"), top1aauc);
CSV.write(joinpath(evalpath, "ranktop1aauc.csv"), ranktop1aauc);
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
rename!(valuedf, :dataset => :test)
push!(valuedf, cat(1,["test auc"],[x[1] for x in colwise(missmean, testauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["train auc"],[x[1] for x in colwise(missmean, trainauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 5%"],[x[1] for x in colwise(missmean, top5auc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 1%"],[x[1] for x in colwise(missmean, top1auc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["mean fit time"],[x[1] for x in colwise(missmean, meanfitt[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["mean predict time"],[x[1] for x in colwise(missmean, meanpredictt[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["test auc - augmented"],[x[1] for x in colwise(missmean, testaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["train auc - augmented"],[x[1] for x in colwise(missmean, trainaauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 5% - augmented"],[x[1] for x in colwise(missmean, top5aauc[[Symbol(alg) for alg in algnames]])]))
push!(valuedf, cat(1,["top 1% - augmented"],[x[1] for x in colwise(missmean, top1aauc[[Symbol(alg) for alg in algnames]])]))
CSV.write(joinpath(evalpath, "valuesummary.csv"), valuedf);

rankeddf = createdf(algnames)
rankeddf = [rankeddf; ranktestauc[end,:]]
rankeddf[:dataset][end] = "test auc"
rankeddf = [rankeddf; ranktrainauc[end,:]]
rankeddf[:dataset][end] = "train auc"
rankeddf = [rankeddf; ranktop5auc[end,:]]
rankeddf[:dataset][end] = "top 5%"
rankeddf = [rankeddf; ranktop1auc[end,:]]
rankeddf[:dataset][end] = "top 1%"
rankeddf = [rankeddf; rankmeanfitt[end,:]]
rankeddf[:dataset][end] = "mean fit time"
rankeddf = [rankeddf; rankmeanpredictt[end,:]]
rankeddf[:dataset][end] = "mean predict time"
rankeddf = [rankeddf; ranktestaauc[end,:]]
rankeddf[:dataset][end] = "test auc - augmented"
rankeddf = [rankeddf; ranktrainaauc[end,:]]
rankeddf[:dataset][end] = "train auc - augmented"
rankeddf = [rankeddf; ranktop5aauc[end,:]]
rankeddf[:dataset][end] = "top 5% - augmented"
rankeddf = [rankeddf; ranktop1aauc[end,:]]
rankeddf[:dataset][end] = "top 1% - augmented"
rename!(rankeddf, :dataset => :test)
CSV.write(joinpath(evalpath, "ranksummary.csv"), rankeddf);

if verb
	println("")
	println("Summary of mean test values:")
	showall(rounddf(valuedf, 2, 2))
	println("")
	println("Summary of mean ranks:")
	showall(rounddf(rankeddf, 2, 2))
	println("")
end
