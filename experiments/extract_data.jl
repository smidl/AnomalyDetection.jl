using FileIO, ValueHistories

#datapath = "/opt/output/anomaly" #master path where data are stored
#datapath = "/home/vit/vyzkum/anomaly_detection/data/aws/anomaly"

#extpath = "/opt/output/extracted"
#extpath = "/home/vit/vyzkum/anomaly_detection/data/aws/extracted"
mkpath(extpath)

datasets = readdir(datapath)
for dataset in datasets
	dpath = joinpath(datapath, dataset)
	algs = readdir(dpath)
	for alg in algs
		apath = joinpath(dpath, alg)
		iters = readdir(apath)
		for iter in iters
			ipatho = joinpath(apath, iter)
			files = readdir(ipatho)

			ipathn = joinpath(extpath, string("$dataset/$alg/$iter"))
			mkpath(ipathn)
			for file in files
				fo = joinpath(ipatho, file)
				fn = joinpath(ipathn, file)
				if isfile(fn) && !rewrite
					nothing
				else
					save(fn,
							"fit_time", load(fo, "fit_time"),
							"predict_time", load(fo, "predict_time"),
							"training_labels", load(fo, "training_labels"),
							"testing_labels", load(fo, "testing_labels"),
							"training_anomaly_score", load(fo, "training_anomaly_score"),
							"testing_anomaly_score", load(fo, "testing_anomaly_score")
							#"params", load(fo, "params")
						)
				end
			end
		end
	end
	println("Extracted data from $(dpath).")
end

