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
        f = figure()
        plot(get(model.history, :discriminator_loss)..., label = "discriminator loss")
        plot(get(model.history, :generator_loss)..., label = "generator loss")
        plot(get(model.history, :reconstruction_error)..., label = "reconstruction error")
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
function plot(model::fmGANmodel)
    # plot model loss
    if model.history == nothing
        println("No data to plot, set tracked = true before training.")
        return
    else
        f = figure()
        plot(get(model.history, :discriminator_loss)..., label = "discriminator loss")
        plot(get(model.history, :generator_loss)..., label = "generator loss")
        plot(get(model.history, :reconstruction_error)..., label = "reconstruction error")
        plot(get(model.history, :feature_matching_loss)..., label = "feature-matching loss")
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
function plot(model::sVAEmodel)
    # plot model loss
    if model.history == nothing
        println("No data to plot, set tracked = true before training.")
        return
    else
        f = figure()
        plot(get(model.history, :discriminator_loss)..., label = "discriminator loss")
        plot(get(model.history, :vae_loss)..., label = "VAE loss")
        plot(get(model.history, :reconstruction_error)..., label = "reconstruction error")
        title("model loss")
        ylabel("loss")
        xlabel("iteration")
        legend()
    end
end

