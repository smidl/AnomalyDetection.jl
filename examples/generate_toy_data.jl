using PyPlot
using JLD
using AnomalyDetection

# generate clusters of 2D data and some outliers
nclusters = 3
M = 2
means = Array{Float64, 2}(nclusters, M)
vars = Array{Float64, 3}(nclusters, M, M)
for n in 1:nclusters
    means[n,:] = rand(Float64,M)
    vars[n,:,:] = diagm(rand(Float64, M))/20
end

clustersize = 30
outlierratio = 0.15
spread = 5
noutliers = Int(floor(clustersize*nclusters*outlierratio))
ninliers = clustersize*nclusters
npoints = noutliers + ninliers
X = Array{Float64}(npoints, M)
labels = Array{Int64}(npoints)

for n in 1:nclusters
    X[(n-1)*clustersize+1:n*clustersize,:] = randn(clustersize, M)*vars[n,:,:] .+ reshape(means[n,:], 1, M)
    labels[(n-1)*clustersize+1:n*clustersize] = 0
end

for n in ninliers+1:npoints
    if nclusters > 1
        X[n,:] = randn(1,M)*eye(M)*maximum(var(means, 1))*spread .+ reshape(mean(means, 1), 1, M)
    else
        X[n,:] = randn(1,M)*eye(M)*maximum(vars)*spread .+ reshape(mean(means, 1), 1, M)
    end
    labels[n] = 1
end
    
figure()
scatter(X[ninliers+1:npoints, 1], X[ninliers+1:npoints, 2])
for n in 1:nclusters
    scatter(X[(n-1)*clustersize+1:n*clustersize,1], X[(n-1)*clustersize+1:n*clustersize,2])
end
show()


# now create a DataSet variable and save it
data = AnomalyDetection.Dataset(X', labels)
save("toy_data_$(nclusters).jld", "data", data)
