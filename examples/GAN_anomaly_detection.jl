
using Plots
plotly()
import Plots: plot
clibrary(:Plots)
using JLD
code_path = "../src/"
push!(LOAD_PATH, code_path)
using AnomalyDetection

# load data
dataset = load("toy_data_3.jld")["data"]
X = AnomalyDetection.Float.(dataset.data)
Y = dataset.labels
nX = X[:, Y.==0]
M, N = size(X)

# GAN settings
zdim = 1 # code dimension
xdim = M # dimension of data
hiddendim = 32  

# setup the GAN model object
gsize = [zdim; hiddendim; hiddendim; xdim] # generator layout
dsize = [xdim; hiddendim*2; hiddendim*2; 1] # discriminator layout
lambda = 0.5 # anomaly score parameter in [0, 1]
# 1 - ignores the discriminator score
# 0- ignores the reconstruction error score
threshold = 0 # classification threshold, is recomputed (getthreshold or when using fit!)
contamination = size(Y[Y.==1],1)/size(Y, 1) # contamination ratio
L = 30 # batchsize
iterations = 15000 # no of iterations
cbit = 5000 # when should output be printed
verbfit = true # if output should be produced
pz = randn # code distribution (rand should also work)
activation = Flux.leakyrelu # should work better than relu
layer = Flux.Dense
rdelta = 1e-5 # stop training after this reconstruction error is achieved
# this parameter is basically useless in the case of GANs
Beta = 1.0 # for automatic threshold computation, in [0, 1] 
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.traindata
model = GANmodel(gsize, dsize, lambda, threshold, contamination, L, iterations, cbit,
    verbfit, pz = pz, activation = activation, rdelta = rdelta, Beta = Beta, 
    layer = layer, tracked = tracked)

# fit the model
Z = AnomalyDetection.getcode(model, size(nX,2))
AnomalyDetection.evalloss(model, nX, Z)
AnomalyDetection.fit!(model, nX)
AnomalyDetection.evalloss(model, nX, Z)

"""
	plot(model)

Plot the model loss.
"""
function plot(model::GANmodel)
	# plot model loss
	if model.history == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
        p = plot(model.history[:discriminator_loss], title = "model loss", 
            label = "discriminator loss", 
            xlabel = "iteration", ylabel = "loss", 
            seriestype = :line, 
            markershape = :none)
        plot!(model.history[:reconstruction_error], label = "reconstruction error",
            seriestype = :line, markershape = :none, title = "model loss")
        plot!(model.history[:generator_loss], label = "generator loss",
            seriestype = :line, markershape = :none, 
            c = :green,
            title = "model loss")
        return p
    end
end


# plot model loss
display(plot(model))

if !isinteractive()
    gui()
end

# generate new data
Xgen = AnomalyDetection.generate(model, N)

# plot them
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
p = scatter(nX[1,:], nX[2,:], title = "discriminator contours",
    xlims = xl, ylims = yl, label = "data")
scatter!(p, Xgen[1,:], Xgen[2,:], label = "generated data", legend = (0.1, 0.8))

x = linspace(xl[1], xl[2], 30)
y = linspace(yl[1], yl[2], 30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.discriminate(model, AnomalyDetection.Float.([x[j], y[i]]))[1]
    end
end
contourf!(x, y, zz, c = :viridis)

display(p)
if !isinteractive()
    gui()
end

# predict labels
AnomalyDetection.setthreshold!(model, X)
tryhat = AnomalyDetection.predict(model, X)

# get all the labels
tryhat, tstyhat, _, _ = AnomalyDetection.rocstats(dataset, dataset, model);

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

plargs = [(fprvec, recvec, "GAN")]
display(plotroc(plargs...))
if !isinteractive()
    gui()
end

# plot EER for different settings of lambda
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, false_negative_rate
n = 21
lvec = linspace(0,1,n)
eervec = zeros(n)
for i in 1:n
    model.lambda = lvec[i]
    AnomalyDetection.setthreshold!(model, X)
    tryhat, tsthat, trroc, tstroc = AnomalyDetection.rocstats(dataset.data, dataset.labels,
        dataset.data, dataset.labels, model, verb = false)
    eervec[i] = (false_positive_rate(tstroc) + false_negative_rate(tstroc))/2
end

p = plot(lvec, eervec, title = "equal error rate vs lambda",
    xlabel = "lambda",
    ylabel = "EER")

display(p)
if !isinteractive()
    gui()
end
