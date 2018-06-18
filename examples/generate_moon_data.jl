using JLD, PyPlot, AnomalyDetection

ndat = 1000
nanom = 10
xv = π/2+randn(ndat)/3
y = sin.(xv)
yn= y + 0.01randn(size(xv))
## normalize data
yn /= 1.2maximum(yn)
y /= 1.2maximum(y)
# add anomalies
aX = randn(2,nanom)/5 .+ [1.5, 0.5]

figure()
scatter(xv,yn)
(size(aX,2) > 0)? scatter(aX[1,:], aX[2,:]) : nothing
xlabel("x")
ylabel("y")
title("Regression")
show()

# create a dataset and save it
X = cat(2, xv, yn)'
labels = zeros(1000)
(size(aX,2) > 0)? (X = cat(2, X, aX); labels = cat(1, labels, ones(nanom))) : nothing
data = AnomalyDetection.Dataset(X, labels)
save("moon.jld", "data", data)
