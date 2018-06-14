The main script to be called from MATLAB is matparams.jl. Call it e.g. as `julia matparams.jl /path/to/inputs.mat /path/to/outputs.mat`. Examples of inputs are in the examples folder. 

You will need the EvalCurves.jl package, install it in julia with `Pkg.clone("https://github.com/vitskvara/EvalCurves.jl.git")`.

Parameter values:

"dataset" - one of ["haberman", "iris", "ecoli", "yeast", "abalone"]. For detailed overview of datasets (dimensions and anomalous/normal data counts) run "julia print_dataset_overviews.jl"

"model" - one of ["AE", "VAE", "GAN", "fmGAN"]. "VAE" and "fmGAN" takes most time to train and fit.

"modelparams" - a Dict() with inputs parameters that can be tuned. Soem are model-dependent.
	
	"nhid" - Integer, [1, Inf]. Number of hidden layers, used in all models. 

	"ldim" - Integer, [1, Inf]. Latent dimension, used in all models. Potentially it should not 	make sense that it is larger than the dimensionality of input data.

	"eta" - Real, [0,Inf]. Learning rate of the optimizer.

	VAE:

	"lambda" - Real, [1e-10, 1e0]. Weight of the KLD in the loss function. Typical values are  
		1e-3, 1e-2, ...

	GAN:

	"lambda" - Real, [0, 1]. Weight of reconstruction error in anomaly score. Typical values are 	0.1, 0.2, ...

	fmGAN:

	"lambda" - Real, [0, 1]. Weight of reconstruction error in anomaly score. Typical values are 	0.1, 0.2, ...

	"alpha" - Real, [0, Inf]. Weight of generator loss in anomaly scoring. Typical values are 
		1e-6, 1e0, 1e6.

