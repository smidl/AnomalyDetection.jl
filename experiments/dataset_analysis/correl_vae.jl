using AnomalyDetection,ValueHistories, EvalCurves
#using Plots
#plotly()
using PyPlot

# get and plot the data
dataset = "breast-cancer-wisconsin"
trData, tstData, c = AnomalyDetection.getdata(dataset)
i,j=1,3
X = cat(2,trData.data,tstData.data)[[i,j],:]
ytrue = cat(1,trData.labels,tstData.labels) 
#figure()
#scatter(X[1,:],X[2,:],c=ytrue)

# now train a VAE on the normal samples
vae = AnomalyDetection.VAE([2,4,4,2],[1,4,4,2])
trx = X[:,ytrue.==0]
h = MVHistory()
AnomalyDetection.fit!(vae,trx,256,iterations=10000, cbit=500,lambda=0.0001,history=h)
l = get(h,:loss)
#figure()
#plot(l[1],l[2])

# get the estimated labels
cont = length(ytrue[ytrue.==1])/length(ytrue)
println("contamination rate: $cont")
tr = AnomalyDetection.getthreshold(vae,X,1,cont)
println("anomaly detection threshold: $tr")
yhat = AnomalyDetection.classify(vae, X,tr,1)

# now construct the anomaly score map
# get limits of the figure
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)

# compute the anomaly score on a grid
x = linspace(xl[1], xl[2], 30)
y = linspace(yl[1], yl[2], 30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.anomalyscore(vae, AnomalyDetection.Float.([x[j], y[i]]), 1)
    end
end

# plot it all
f = figure()
contourf(x, y, zz)
scatter(X[1, ytrue.==1], X[2, ytrue.==1], c = "r", label = "true positive")
scatter(X[1, ytrue.==0], X[2, ytrue.==0], c = "g", label = "true negative")
scatter(X[1, yhat.==1], X[2, yhat.==1], c = "b", s=5, label = "predicted positive")
scatter(X[1, yhat.==0], X[2, yhat.==0], c = "y", s=5, label = "predicted negative")
title("classification results - $dataset")
xlim(xl)
ylim(yl)
legend()
#savefig("contour_plot.eps")

# also, get the anomaly score for the samples
as = AnomalyDetection.anomalyscore(vae, X, 1)
a,b = EvalCurves.roccurve(as, ytrue)
auc = EvalCurves.auc(a,b)
EvalCurves.plotroc((a,b,"AUC = $auc"))
#savefig("auroc.eps")
show()