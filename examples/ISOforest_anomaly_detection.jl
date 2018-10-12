
using PyPlot, FileIO, AnomalyDetection, EvalCurves
import PyPlot: plot
using ScikitLearn
using ScikitLearn: @sk_import, fit!, predict
using ScikitLearn.Utils: meshgrid 

dataset = load("toy_data_3.jld2")["data"]

figure()
X = dataset.data
Y = dataset.labels
scatter(X[1, Y.==1], X[2, Y.==1])
scatter(X[1, Y.==0], X[2, Y.==0])
show()

# import the isolation forest from SKlearn
@sk_import ensemble: IsolationForest

n_estimators=100  # how many estimators to use
max_samples="auto" # how many samples to draw from X for each estimator
contamination = 0.0 # size(Y[Y.==1],1)/size(Y,1) # to set the decision threshold
max_features=1.0 # how many features to use (if float, then it is a ratio)
bootstrap=false # bootstrapping - if false, sample without replacing
n_jobs=1 # how many cores to use
#random_state=None # seed or generator
verbose=0 # verbosity of the fitting
isoforest = IsolationForest(n_estimators, max_samples, contamination, max_features, bootstrap,
    n_jobs, verbose)

fit!(isoforest, X[:,Y.==0]')

import ScikitLearn: decision_function

decision_function(isoforest, X')[:]

tsthat = AnomalyDetection.labels2bin(predict(isoforest, dataset.data'))

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
        zz[i,j] = ScikitLearn.decision_function(isoforest, [x[j], y[i]]')[1]
    end
end

# plot it all
f = figure()
contourf(x, y, zz)
scatter(X[1, tsthat.==0], X[2, tsthat.==0], c = "r", 
    label = "predicted positive")
scatter(X[1, tsthat.==1], X[2, tsthat.==1], c = "g", 
    label = "predicted negative")
title("classification results")
xlim(xl)
ylim(yl)
legend()
show()


# plot ROC curve and compute AUROC score
ascore = 1.0.-decision_function(isoforest, X')[:]
fprvec, tprvec = EvalCurves.roccurve(ascore, Y)
auroc = round(EvalCurves.auc(fprvec, tprvec),digits=3)
EvalCurves.plotroc((fprvec, tprvec, "AUROC = $(auroc)"))
show()
