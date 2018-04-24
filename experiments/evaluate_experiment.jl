include("./eval.jl")

data_path = "./data"
outpath = "./output"
mkpath(outpath)

algnames = ["kNN", "AE", "VAE", "sVAE", "GAN", "fmGAN"]

# create a dataframe with a row for each experiment
println("Computing basic experiment statistics...")
@time allthedata = computestats(data_path,algnames);
writetable(joinpath(outpath, "allthedata.csv"), allthedata)
println("Done.")

# first experiment
println("First experiment - rank algorithms on maximum auroc in an iteration, average over iterations.")
println("")
maxauc = maxauroc(allthedata, algnames);
rankmaxauc = rankdf(maxauc);
writetable(joinpath(outpath, "maxauc.csv"), maxauc);
writetable(joinpath(outpath, "rankmaxauc.csv"), rankmaxauc);
showall(maxauc)
println("")
showall(rankmaxauc)
println("")