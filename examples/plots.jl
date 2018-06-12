using PyPlot, AnomalyDetection
import PyPlot: plot

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
        f = figure()
        plot(get(model.history, :loss)..., label = "loss")
        plot(get(model.history, :reconstruction_error)..., 
            label = "reconstruction error")
        plot(get(model.history, :KLD)..., label = "KLD")
        title("model loss")
        ylabel("loss + reconstruction error")
        xlabel("iteration")
        legend()
    end
end
