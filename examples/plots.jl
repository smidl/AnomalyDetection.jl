using PyPlot, AnomalyDetection
import PyPlot: plot

"""
    plot(AEmodel)

Plot the model loss.
"""
function plot(model::AEmodel)
    # plot model loss
    if model.history == nothing
        println("No data to plot, set tracked = true before training.")
        return
    else
        f = figure()
        plot(get(model.history, :loss)..., label = "loss")
        title("model loss")
        ylabel("loss")
        xlabel("iteration")
        legend()
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
        f = figure()
        plot(get(model.history, :loss)..., label = "loss")
        plot(get(model.history, :likelihood)..., 
            label = "likelihood")
        plot(get(model.history, :KLD)..., label = "KLD")
        title("model loss")
        ylabel("loss + likelihood")
        xlabel("iteration")
        legend()
    end
end
