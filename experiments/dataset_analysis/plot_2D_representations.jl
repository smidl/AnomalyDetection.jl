# run as 'julia plot_2D_representations.jl dataset'
# overview of datasets can be obtained running 'julia ../print_loda_overview.jl'
(length(ARGS) < 1)? 
	error("Provide the dataset name when running the script, e.g. 'julia plot_2D_representations.jl haberman.") :
	dataset = ARGS[1]

fp = @__DIR__
include(joinpath(fp, "utils.jl"))

pca_path = joinpath(fp,"pca_2D-data")
tsne_path = joinpath(fp,"./tsne_2D-data")
pca_data = Basicset(joinpath(pca_path,dataset))
tsne_data = Basicset(joinpath(tsne_path,dataset))

scatter_data(pca_data,"$dataset - PCA"; s = 10)
scatter_data(tsne_data,"$dataset - tSne"; s = 10)
show()