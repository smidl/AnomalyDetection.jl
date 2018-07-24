# run as 'julia show_ffs_res.jl all vae 10' or 'julia show_ffs_res.jl some knn 30'
# julia show_ffs_res.jl which_anomalies which_algorithm_leads no_lines
include("../eval.jl")
data = ARGS[1]
var = ARGS[2]
nlines = Int(parse(ARGS[3]))

function getall(datapath)
    fs = readdir(datapath)
    df = loadtable(joinpath(ffsp,fs[1]),5)
    df[:dataset] = split(fs[1],".")[1]
    for f in fs[2:end]
        _df = loadtable(joinpath(ffsp,f),5)
        _df[:dataset] = split(f,".")[1]
        df = vcat(df,_df)
    end
    df = hcat(df[end],df[1:end-1])
    rename!(df,:x1 => :dataset)
    return df
end

function diff(df)
    df[:diff] = df[:knn] - df[:vae];
    df[:absdiff] = abs.(df[:diff]);
    return df
end

sdf = CSV.read("20_some.csv")
adf = CSV.read("20_some.csv")

if data == "all"
	if var == "knn"
		df = sort(adf[adf[:diff].>0,:],:absdiff,rev=true)
	else
		df = sort(adf[adf[:diff].<0,:],:absdiff,rev=true)
	end
else
	if var == "knn"
		df = sort(sdf[sdf[:diff].>0,:],:absdiff,rev=true)
	else
		df = sort(sdf[sdf[:diff].<0,:],:absdiff,rev=true)
	end
end
showall(df[1:nlines,:])