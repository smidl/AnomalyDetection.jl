include("./eval.jl")

#data_path = "./data"
#outpath = "./output"
#evalpath = "./eval"
data_path = "/home/vit/vyzkum/anomaly_detection/data/aws/anomaly"
outpath = "/home/vit/vyzkum/anomaly_detection/data/aws/output"
evalpath = "/home/vit/vyzkum/anomaly_detection/data/aws/eval"
datasets = filter(s->s!="persistant-connection",readdir(data_path))
mkpath(outpath)
mkpath(evalpath)

#experiment = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1:5

algnames = ["kNN", "IsoForest", "AE", "VAE", "sVAE", "GAN", "fmGAN"]

# create a dataframe with a row for each experiment
println("Computing basic experiment statistics...")
for dataset in datasets
	f = joinpath(outpath, "$(dataset).csv")
	if !isfile(f)
		df = computedatasetstats(data_path, dataset, algnames)
		writetable(f, df)
		println("Computed and saved $(f)!")
	else
		println("$f is already present!")
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
topauc = collectscores(outpath, algnames, topprec)
ranktopauc = rankdf(topauc)
writetable(joinpath(evalpath, "topauc.csv"), topauc);
writetable(joinpath(evalpath, "ranktopauc.csv"), ranktopauc);
showall(topauc)
println("")
showall(ranktopauc)
println("")

# fourth experiment
println("Fourth experiment - rank by mean fit time.")
println("")
meanfitt = collectscores(outpath, algnames, (x,y)->meantime(x,y,"fit_time"))
showall(meanfitt)
println("")
rankmeanfitt = rankdf(meanfitt, false)
writetable(joinpath(evalpath, "meanfitt.csv"), meanfitt);
writetable(joinpath(evalpath, "rankmeanfitt.csv"), rankmeanfitt);
showall(rankmeanfitt)
println("")

# fifth experiment
println("Fifth experiment - rank by mean predict time.")
println("")
meanpredictt = collectscores(outpath, algnames, (x,y)->meantime(x,y,"predict_time"))
rankmeanpredictt = rankdf(meanpredictt, false)
writetable(joinpath(evalpath, "meanpredictt.csv"), meanpredictt);
writetable(joinpath(evalpath, "rankmeanpredictt.csv"), rankmeanpredictt);
showall(meanpredictt)
println("")
showall(rankmeanpredictt)
println("")
