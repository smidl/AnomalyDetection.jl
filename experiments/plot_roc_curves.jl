using JLD
using PyPlot
push!(LOAD_PATH,"../src")
using AnomalyDetection

function plot_roc_curves(data_path)
	figure()
	plot(linspace(0,1,100), linspace(0,1,100), "--", c = "gray", alpha = 0.5)
	xlim([0, 1])
	ylim([0, 1])
	xlabel("false positive rate")
	ylabel("true positive rate")

	# isolation forest
	try
		isoforest_data = load(joinpath(data_path, "ISOforest_roccurve.jld"))
		plot(isoforest_data["false_positive_rate"], isoforest_data["true_positive_rate"],
			lw = 1, label = "ISOforest")
		scatter(isoforest_data["false_positive_rate"], isoforest_data["true_positive_rate"],
			s = 5)
	end

	# autoencoder
	try
		ae_data = load(joinpath(data_path, "AE_roccurve.jld"))
		plot(ae_data["false_positive_rate"], ae_data["true_positive_rate"],
			lw = 1, label = "AE")
		scatter(ae_data["false_positive_rate"], ae_data["true_positive_rate"],
			s = 5)
	end

	# variational autoencoder
	try
		vae_data = load(joinpath(data_path, "VAE_roccurve.jld"))
		plot(vae_data["false_positive_rate"], vae_data["true_positive_rate"],
			lw = 1, label = "VAE")
		scatter(vae_data["false_positive_rate"], vae_data["true_positive_rate"],
			s = 5)
	end

	# GAN
	try
		gan_data = load(joinpath(data_path, "GAN_roccurve.jld"))
		plot(gan_data["false_positive_rate"], gan_data["true_positive_rate"],
			lw = 1, label = "GAN")
		scatter(gan_data["false_positive_rate"], gan_data["true_positive_rate"],
			s = 5)
	end

	# fmGAN
	try
		fmgan_data = load(joinpath(data_path, "fmGAN_roccurve.jld"))
		plot(fmgan_data["false_positive_rate"], fmgan_data["true_positive_rate"],
			lw = 1, label = "fmGAN")
		scatter(fmgan_data["false_positive_rate"], fmgan_data["true_positive_rate"],
			s = 5)
	end

	# symetric variational autoencoder
	try
		svae_data = load(joinpath(data_path, "sVAE_roccurve.jld"))
		plot(svae_data["false_positive_rate"], svae_data["true_positive_rate"],
			lw = 1, label = "sVAE")
		scatter(svae_data["false_positive_rate"], svae_data["true_positive_rate"],
			s = 5)
	end

	legend( loc = "lower right")
	show()
end

if size(ARGS,1) > 0
	data_path = ARGS[1]
else
	data_path = "./data/magic-telescope_0.8_easy_0.02_low/"
end

plot_roc_curves(data_path)