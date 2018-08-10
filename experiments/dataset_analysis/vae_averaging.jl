# julia vae_averaging.jl dataset seed
dataset = ARGS[1]
seed = Int(parse(ARGS[2]))

using AnomalyDetection, PyPlot, EvalCurves, FileIO, ProgressMeter

include("utils.jl")

outpath = "vae_averaging"
inpath = "tsne_2D-data"

data = getdata(dataset, false,seed,loc=inpath)
Nvaes = 100
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
println("Training $(Nvaes) VAEs.")
vaes = trainvaes(data,Nvaes);
println("Done.")

# get a very precise AUC curve
println("Computing a precise AUC curve.")
Ns = 1:Nvaes
aucs = []
M = 1
for n in Ns
    as = averageanomalyscore(vaes[1:n],tstdata.data,M)
    auc = EvalCurves.auc(EvalCurves.roccurve(as,ytrue)...)
    push!(aucs,auc)
end
println("Done.")

# save output
save(joinpath(outpath,"$(dataset)_$(seed)_$(Nvaes)_auc.jld"), "auc", aucs, "N", Ns)

# produce a plot
figure()
plot(Ns,aucs)
#plot(Ns, ones(N)*auc_knn, label = "knn result")
title("$(dataset)_$(seed), AUC of VAE averaging")
xlabel("no of models")
ylabel("AUC")
savefig(joinpath(outpath,"$(dataset)_$(seed)_$(Nvaes)_auc.png"))

# try this with different sortings of the vaelist
println("Computing multiple randomized AUC curves.")
Nex = 10
experiments = []
N = length(vaes)
Nssp = 1:5:Nvaes
M = 1
y = tstdata.labels;
p = Progress(Nex,1)
for experiment in 1:Nex
    aucssp = []
    vaes_sorted = vaes[sample(1:Nvaes,Nvaes,replace=false)]
    for n in Nssp
        as = averageanomalyscore(vaes_sorted[1:n],tstdata.data,M)
        auc = EvalCurves.auc(EvalCurves.roccurve(as,ytrue)...)
        push!(aucssp,auc)
    end
    push!(experiments, aucssp)
    next!(p)
end
println("Done.")

# plot it
figure()
plot(Ns,aucs)
for aucssp in experiments
    plot(Nssp, aucssp)
end
plot(Ns, ones(Nvaes)*auc_knn, "--", label = "knn result", c= "k")
title("$(dataset)_$(seed), AUC of VAE averaging, multiple runs")
xlabel("no of models")
legend()
ylabel("AUC")
savefig(joinpath(outpath,"$(dataset)_$(seed)_$(Nvaes)_auc_$(Nex)reruns.png"))

save(joinpath(outpath,"$(dataset)_$(seed)_$(Nvaes)_auc.jld"), "auc", aucs, "N", Ns, 
    "N_sparse", Nssp, "auc_sparse_experiment", experiments)

# now create a grid of as contours
function asgrid(models, data, M, griddensity)
    xl,yl = xylims(data)
    x = linspace(xl[1], xl[2], griddensity)
    y = linspace(yl[1], yl[2], griddensity)
    zz = zeros(size(y,1),size(x,1))
    for i in 1:size(y, 1)
        for j in 1:size(x, 1)
            zz[i,j] = averageanomalyscore(models, AnomalyDetection.Float.([x[j], y[i]]), M)
        end
    end
    return x,y,zz
end

function vaescontours(models, N, data, M, griddensity; t="")
    x,y,zz = asgrid(models[1:N], data, M , griddensity)
    contourf(x,y,zz,50)
    colorbar()
    title(t)
end

# now do the same plot for multiple different models
println("Producing a grid plot of anomaly score contours.")
f = figure(figsize=(20,15))
M = 1 # number of samplings - for such a low lambda used (1e-5), this is enough
gd = 30 # gridsize
pNs = [1,2,3,5,15,30,50,75,Nvaes]
for (i,n) in enumerate(pNs)
    subplot(330+i)
    t = "$(dataset)_$(seed), AS contours of $n models\ntraining data"
    vaescontours(vaes, n, data, M, gd, t = t)
    scatter(trdata.data[1,:],trdata.data[2,:], c = "g", s = 5)
end
savefig(joinpath(outpath,"$(dataset)_$(seed)_contour_grid.png"))
println("Done.")