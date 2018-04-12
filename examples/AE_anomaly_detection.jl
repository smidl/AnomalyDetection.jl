
using Plots
plotly()
import Plots: plot
using JLD
using ScikitLearn: @sk_import, fit!, predict
using ScikitLearn.Utils: meshgrid 

code_path = "../src/"

push!(LOAD_PATH, code_path)
using AnomalyDetection

dataset = load("toy_data_3.jld")["data"]

X = dataset.data
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
contamination = size(Y[Y.==1],1)/size(Y[Y.==0],1) # to set the decision threshold
iterations = 5000
cbit = 1000 # when callback is printed
verbfit = true 
activation = Flux.relu
rdelta = 0.002 # reconstruction error threshold when training is stopped
Beta = 1.0 # for automatic threshold computation, in [0, 1] 
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.traindata

# model might have to be restarted if loss is > 0.01
model = AEmodel(esize, dsize, L, threshold, contamination,
    iterations, cbit, verbfit, activation = activation, rdelta = rdelta, 
    tracked = tracked)

AnomalyDetection.fit!(model, X, Y)
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
        display(p)
    end
end

#plot(model)
plot(model.history[:loss], title = "model loss", label = "loss", 
            xlabel = "iteration", ylabel = "loss", seriestype = :line, 
            markershape = :none)	
gui()

#        display(p)
model(nX)

nX

model(X)

X

# predict labels
yhat = AnomalyDetection.predict(model, X)

# training data = testing data
# this outputs labels
tryhat, tsthat, _, _ = AnomalyDetection.rocstats(dataset, dataset, model);
