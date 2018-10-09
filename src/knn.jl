import NearestNeighbors: KDTree, knn

"""
kNN model structure for anomaly detection. Uses KDtree for distance computation. 
Anomaly score is computed as an average of distances to k nearest neighbours.
"""
mutable struct kNN
    k
    kdtree
    metric # distance metric
    distances # all/last
    contamination # contamination rate
    threshold # decision threshold for classification
    reduced_dim # if this is true, reduce dimensionality of data if > 10 (PCA)
    PCA # tuple containing (P, mu) of the PCA
    fitted
    Beta::Float
end

"""
   kNN(k, contamination, [metric, distances, reduced_dim, threshold, Beta])

kNN model constructor. 

k - number of nearest neighbours taken into consideration
contamination - ratior of contaminated to all samples
metric - for distance computation, using Distances package
distances - use average of all k-nearest or just the kth-nearest neighbour distance
reduced_dim - if this is true, for problems with dim > 10 this is reduced with PCA
threshold - classification threshold
Beta - for automatic threshold computation
"""
function kNN(k::Int, contamination::Real; metric = Euclidean(), distances = "all", 
        threshold = 0.0, reduced_dim = false, Beta = 1.0)
    @assert distances in ["all", "last"]
    return kNN(k, Array{Float,2}(0,0), metric, distances, contamination, threshold,
        reduced_dim, (Array{Float,2}(0,0), Array{Float,1}(0)), false, Beta)
end

"""
    fit!(knn, X)

Fit method for kNN.
"""
function fit!(model::kNN, X)
    X=Float.(X)
    if model.reduced_dim
        m = fit(PCA, X, maxoutdim = 10)
        model.kdtree = KDTree(transform(m,X))
        model.PCA = (projection(m), mean(m))
    else
        model.kdtree = KDTree(X)
    end
    model.fitted = true
    if size(X,2) < model.k
        warn("k is higher than data dimension, setting lower...")
        model.k = size(X,2)
    end
end

"""
    anomalyscore(knn, X, k)

Computes the anomaly score for X using k nearest neighbours.
"""
function anomalyscore(model::kNN, X::Array{Float,2}, k)
    if !model.fitted
        error("Call fit!(model, X, Y) before predict can be used!")
    end
    
    # if needed, perform the PCA
    if model.reduced_dim
        _X = model.PCA[1]'*(X .- model.PCA[2])
    else
        _X = X
    end

    M, N = size(_X)
    
    # now create the output vector of labels
    ascore = Array{Float, 1}(N)
    for n in 1:N
        _, dnk = knn(model.kdtree, _X[:,n], model.k)
        if model.distances == "last"
            ascore[n] = dnk[end]
        else
            ascore[n] = mean(dnk)
        end
    end
    
    return ascore
end
anomalyscore(model::kNN, x::Array{Float, 1}, k) = anomalyscore(model, reshape(x, size(x,1), 1), k)[1]
anomalyscore(model::kNN, X::Union{Array{T, 1},Array{T, 2}} where T<:Real, k) = 
    anomalyscore(model,Float.(X),k)

"""
    anomalyscore(knn, X)

Computes the anomaly score for X.
"""
anomalyscore(model::kNN, X) = anomalyscore(model, X, model.k)

"""
    classify(knn, x, threshold)

Classify an instance x using the discriminator and a threshold.
"""
classify(model::kNN, X) = Int.(anomalyscore(model, X) .> model.threshold)

"""
    setthreshold!(knn, X)

Set model classification threshold based ratior of labels in Y.
"""
function setthreshold!(model::kNN, X)
    model.threshold = getthreshold(model, X, model.contamination; Beta = model.Beta)
end

"""
    predict(knn, X)

Predict labels of X.
"""
function predict(model::kNN, X)
    return classify(model, X)
end
