
using PyPlot, FileIO, AnomalyDetection, EvalCurves, Flux, Statistics
import PyPlot: plot
include("./plots.jl")

# load data
dataset = load("toy_data_3.jld2")["data"]
#dataset = load("moon.jld")["data"]
X = AnomalyDetection.Float.(dataset.data)
Y = dataset.labels
nX = X[:, Y.==0]

# model settings
indim = size(X,1)
hiddendim = 10
latentdim = 2
nlayers = 2
contamination = size(Y[Y.==1],1)/size(Y, 1) # for automatic threshold computation
iterations = 2000
lambda = 1e-3
esize = [indim; hiddendim; hiddendim; latentdim*2] # encoder architecture
#esize = [latentdim; hiddendim; hiddendim; indim]
# decoder architecture
dsize = [latentdim; hiddendim; hiddendim; indim]
#dsize = [indim; hiddendim; hiddendim; 1]
batchsize = 30

# setup the ensemble object
constructor = VAEmodel
nmodels = 20
af = mean # aggregation function for the ensemble
model = Ensemble(constructor, nmodels, af, esize, dsize; contamination = contamination,
    iterations = iterations, verbfit = false, batchsize = batchsize);

# fit the model
@time AnomalyDetection.fit!(model, nX)
AnomalyDetection.setthreshold!(model, X)

# predict labels on testing data
global tryhat = AnomalyDetection.predict(model, X)

# get the labels and roc stats
tryhat, tstyhat = AnomalyDetection.rocstats(dataset, dataset, model);

# anomaly score contour plot
# get limits of the figure
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)

# compute the anomaly score on a grid
x = range(xl[1], stop=xl[2], length=30)
y = range(yl[1], stop=yl[2], length=30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
    end
end

# plot it all
f = figure()
contourf(x, y, zz,30)
scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = "r", 
    label = "predicted positive")
scatter(X[1, tryhat.==0], X[2, tryhat.==0], c = "g", 
    label = "predicted negative")
title("classification results")
xlim(xl)
ylim(yl)
legend()
show()

# plot ROC curve and compute AUROC score
ascore = AnomalyDetection.anomalyscore(model, X);
fprvec, tprvec = EvalCurves.roccurve(ascore, Y)
auroc = round(EvalCurves.auc(fprvec, tprvec),digits=3)
EvalCurves.plotroc((fprvec, tprvec, "AUROC = $(auroc)"))
show()
