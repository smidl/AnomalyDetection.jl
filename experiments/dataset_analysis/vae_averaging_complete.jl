# julia vae_averaging_complete.jl dataset seed
dataset = ARGS[1]
seed = Int(parse(ARGS[2]))

using AnomalyDetection, PyPlot, EvalCurves, FileIO, ProgressMeter

include("utils.jl")

outpath = "vae_averaging"
inpath = "tsne_2D-data"

data = getdata(dataset, false,seed,loc=inpath)
Nvaes = 50
Nruns = 10
trdata = data[1]
tstdata = data[2]
ytrue = tstdata.labels

function trainvaes(data,N::Int)
    trdata = data[1]
    tstdata = data[2]
    models = []
    p = Progress(N,1)
    for n in 1:N
        va, model, _ = vaescore(trdata, tstdata)
        push!(models,model)
        next!(p)
    end
    return models
end

averageanomalyscore(models,X,M) = mean([AnomalyDetection.anomalyscore(models[i],X,M) for i in 1:length(models)])

# get the knn score
knnres = knnscore(trdata, tstdata, 1)
auc_knn = knnres[1]

# train all the vaes
aucruves = []
for run in 1:Nruns
    println("Training $(Nvaes) VAEs, iteration $(run)/$(Nruns).")
    vaes = trainvaes(data,81);
    println("Done.")

    # get a very precise AUC curve
    println("Computing a precise AUC curve, iteration $(run)/$(Nruns).")
    N = length(vaes)
    Ns = 1:N
    aucs = []
    M = 1
    for n in Ns
        as = averageanomalyscore(vaes[1:n],tstdata.data,M)
        auc = EvalCurves.auc(EvalCurves.roccurve(as,ytrue)...)
        push!(aucs,auc)
    end
    push!(aucruves, aucs)
    println("Done.")
end

# save output
save(joinpath(outpath,"$(dataset)_$(seed)_auc_N_complete.jld"), "auc_curves", auccurves, "N", Ns)

# produce a plot
figure()
for aucs in auccurves
    plot(Ns,aucs)
end
plot(Ns, ones(N)*auc_knn, label = "knn result")
title("$(dataset)_$(seed), AUC of VAE averaging, complete reruns")
xlabel("no of models")
ylabel("AUC")
savefig(joinpath(outpath,"$(dataset)_$(seed)_auc_N_complete.png"))

