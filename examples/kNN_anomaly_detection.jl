
using PyPlot, FileIO, AnomalyDetection, EvalCurves, Distances
import PyPlot: plot
include("./plots.jl")

dataset = load("toy_data_3.jld2")["data"]

X = AnomalyDetection.Float.(dataset.data);
Y = dataset.labels;
nX = X[:, Y.==0]

# model parameters
k = 11 # number of nearest neighbors
metric = Distances.Euclidean() # any of metric from Distance package
distances = "all" # "all"/"last" - use average of all or just the k-th nearest neighbour
contamination = size(Y[Y.==1],1)/size(Y,1)
reduced_dim = false # if dim > 10, use PCA to reduce it
Beta = 1.0
#model = kNN(k, metric = metric, weights = weights, reduced_dim = reduced_dim)
model = kNN(k, contamination, metric = metric, distances = distances,
    reduced_dim = reduced_dim, Beta = Beta)

size(nX)

AnomalyDetection.fit!(model, nX);
AnomalyDetection.setthreshold!(model, X);

AnomalyDetection.anomalyscore(model, X)

# this fits the model and produces predicted labels
tryhat, tstyhat = AnomalyDetection.rocstats(X, Y, X, Y, model);

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
contourf(x, y, zz)
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
