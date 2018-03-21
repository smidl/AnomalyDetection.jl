
using PyPlot
using JLD
using ScikitLearn: @sk_import, fit!, predict
using ScikitLearn.Utils: meshgrid 

code_path = "../src/"

push!(LOAD_PATH, code_path)
using AnomalyDetection

dataset = load("toy_data_3.jld")["data"]

figure()
X = dataset.data
y = dataset.labels
scatter(X[1, y.==1], X[2, y.==1])
scatter(X[1, y.==0], X[2, y.==0])
show()

# import the isolation forest from SKlearn
@sk_import ensemble: IsolationForest

n_estimators=100  # how many estimators to use
max_samples="auto" # how many samples to draw from X for each estimator
contamination = size(y[y.==1],1)/size(y[y.==0],1) # to set the decision threshold
max_features=1.0 # how many features to use (if float, then it is a ratio)
bootstrap=false # bootstrapping - if false, sample without replacing
n_jobs=1 # how many cores to use
#random_state=None # seed or generator
verbose=0 # verbosity of the fitting
isoforest = IsolationForest(n_estimators, max_samples, contamination, max_features, bootstrap,
    n_jobs, verbose)

tryhat, tsthat = AnomalyDetection.quickvalidate!(dataset, dataset, isoforest);

# plot heatmap of the fit
figure()
scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = "r")
ax = gca()
ylim = ax[:get_ylim]()
xlim = ax[:get_xlim]()
xx, yy = meshgrid(linspace(xlim[1], xlim[2], 30), linspace(ylim[1], ylim[2], 30))
zz = zeros(size(xx))
for i in 1:size(xx, 1)
    for j in 1:size(xx, 2)
        zz[i,j] = ScikitLearn.decision_function(isoforest, [xx[i,j], yy[i,j]]')[1]
    end
end
axsurf = ax[:contourf](xx, yy, zz)
cb = colorbar(axsurf, fraction = 0.05, shrink = 0.5, pad = 0.1)
scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = "r", label = "predicted positive")
scatter(X[1, tsthat.==0], X[2, tsthat.==0], c = "g", label = "predicted negative")
show()
