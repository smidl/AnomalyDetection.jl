
using Plots
plotly()
import Plots: plot
clibrary(:Plots)
using JLD

code_path = "../src/"

push!(LOAD_PATH, code_path)
using AnomalyDetection

dataset = load("toy_data_3.jld")["data"]

X = AnomalyDetection.Float.(dataset.data)
Y = dataset.labels
nX = X[:, Y.==0]

# set problem dimensions
indim = size(X,1)
hiddendim = 4
latentdim = 2
nlayers = 3

# model constructor parameters
esize = [indim; hiddendim; hiddendim; latentdim]; # encoder architecture
dsize = [latentdim; hiddendim; hiddendim; indim]; # decoder architecture
L = 30 # batchsize
threshold = 0 # classification threshold, is recomputed when calling fit!
contamination = size(Y[Y.==1],1)/size(Y,1) # to set the decision threshold
iterations = 5000
cbit = 1000 # when callback is printed
verbfit = true 
activation = Flux.relu
rdelta = 0.002 # reconstruction error threshold when training is stopped
Beta = 1.0 # for automatic threshold computation, in [0, 1] 
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.history

# model might have to be restarted if loss is > 0.01
model = AEmodel(esize, dsize, L, threshold, contamination,
    iterations, cbit, verbfit, activation = activation, rdelta = rdelta, 
    tracked = tracked)

AnomalyDetection.fit!(model, nX)
AnomalyDetection.setthreshold!(model, X)
AnomalyDetection.evalloss(model, nX)

"""
	plot(model)

Plot the model loss.
"""
function plot(model::AEmodel)
	# plot model loss
	if model.history == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
        p = plot(model.history[:loss], title = "model loss", label = "loss", 
            xlabel = "iteration", ylabel = "loss", seriestype = :line, 
            markershape = :none)
        return p
    end
end

display(plot(model))
if !isinteractive()
    gui()
end

model(nX)

nX

model(X)

X

# predict labels
yhat = AnomalyDetection.predict(model, X)

# training data = testing data
# this outputs labels
tryhat, tsthat, _, _ = AnomalyDetection.rocstats(dataset, dataset, model);

# plot heatmap of the fit
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
p = scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = :red, label = "predicted positive",
    xlims=xl, ylims = yl, title = "classification results")
scatter!(X[1, tryhat.==0], X[2, tryhat.==0], c = :green, label = "predicted negative",
    legend = (0.7, 0.7))

x = linspace(xl[1], xl[2], 30)
y = linspace(yl[1], yl[2], 30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
    end
end
contourf!(x, y, zz, c = :viridis)

display(p)
if !isinteractive()
    gui()
end

# what are the codes?
z1 = model.ae.encoder(X[:,1:30]).data
z2 = model.ae.encoder(X[:,31:60]).data
z3 = model.ae.encoder(X[:,61:90]).data
za = model.ae.encoder(X[:,91:end]).data

p = scatter(z1[1,:], z1[2,:], label = "first cluster", title = "code distribution")
scatter!(z2[1,:], z2[2,:], label = "second cluster")
scatter!(z3[1,:], z3[2,:], label = "third cluster")
scatter!(za[1,:], za[2,:], markersize = 3, label = "anomalous data")

display(p)
if !isinteractive()
    gui()
end

# plot the roc curve as well
ascore = AnomalyDetection.anomalyscore(model, X);
recvec, fprvec = AnomalyDetection.getroccurve(ascore, Y)

function plotroc(args...)
    # plot the diagonal line
    p = plot(linspace(0,1,100), linspace(0,1,100), c = :gray, alpha = 0.5, xlim = [0,1],
    ylim = [0,1], label = "", xlabel = "false positive rate", ylabel = "true positive rate",
    title = "ROC")
    for arg in args
        plot!(arg[1], arg[2], label = arg[3], lw = 2)
    end
    return p
end

plargs = [(fprvec, recvec, "AE")]
display(plotroc(plargs...))
if !isinteractive()
    gui()
end
