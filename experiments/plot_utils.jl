# various plotting methods
using Plots
plotly()
import Plots: plot
clibrary(:Plots)
push!(LOAD_PATH, "../src")
using AnomalyDetection

"""
	plotroc(args...)

Plot roc curves, where args is an iterable of triples (fprate, tprate, label).
"""
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

"""
	plot(model)

Plot the model loss.
"""
function plot(model::VAEmodel)
	# plot model loss
	if model.history == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
        p = plot(model.history[:loss], title = "model loss", label = "loss", 
            xlabel = "iteration", ylabel = "loss + reconstruction error", 
            seriestype = :line, 
            markershape = :none)
        plot!(model.history[:reconstruction_error], label = "reconstruction error",
            seriestype = :line, markershape = :none, title = "model loss")
        plot!(model.history[:KLD], label = "KLD",
            seriestype = :line, markershape = :none, 
            c = :green,
            title = "model loss")
        return p
    end
end

"""
	plot(model)

Plot the model loss.
"""
function plot(model::sVAEmodel)
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
        plot!(model.history[:vae_loss], label = "VAE loss",
            seriestype = :line, markershape = :none)
        plot!(model.history[:reconstruction_error], label = "reconstruction error",
            seriestype = :line, markershape = :none, 
            c = :green,
            title = "model loss")
        return p
    end
end

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

"""
	plot(model)

Plot the model loss.
"""
function plot(model::fmGANmodel)
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
        plot!(model.history[:feature_matching_loss], label = "feature-matching loss",
            seriestype = :line, markershape = :none, title = "model loss")
        return p
    end
end

"""
	plot2Dclassandsurf(model, X)

Plot classification labels and anomaly score contours for 2D input data X.
"""
function plot2Dclassandsurf(model, X)
	yhat = AnomalyDetection.predict(model, X)
	xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
	yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
	p = scatter(X[1, yhat.==1], X[2, yhat.==1], c = :red, label = "predicted positive",
	    xlims=xl, ylims = yl, title = "classification results")
	scatter!(p, X[1, yhat.==0], X[2, yhat.==0], c = :green, label = "predicted negative",
	    legend = (0.7, 0.7))

	x = linspace(xl[1], xl[2], 30)
	y = linspace(yl[1], yl[2], 30)
	zz = zeros(size(y,1),size(x,1))
	for i in 1:size(y, 1)
	    for j in 1:size(x, 1)
	        zz[i,j] = AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
	    end
	end
	contourf!(p, x, y, zz, c = :viridis)
	return p
end
