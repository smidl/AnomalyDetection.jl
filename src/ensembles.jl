mutable struct Ensemble
	models::Vector
	N::Int
	agregf
	contamination::Real
	threshold::Real
	Beta::Real
end

"""
	Ensemble(constructor, N, agregf, [args...; contamination, Beta, kwargs...])

Universal ensemble constructor. Arguments 'args' and 'kwargs' are arguments 
of the model constructor.

constructor - a model of choice, e.g. AnomalyDetection.VAEmodel
\nN - number of models in ensemble
\nagregf - aggregating function, e.g. mean
\ncontamination [0.0] - percentage of anomalous samples in all data for automatic threshold computation
\nBeta [1.0] - how tight around normal data is the automatically computed threshold
"""
function Ensemble(constructor, N::Int, agregf, args...; contamination = 0.0, Beta = 1.0, kwargs...)
	models = [constructor(args...; kwargs...) for n in 1:N]
	return Ensemble(models, N, agregf,contamination,0.0,Beta)
end

"""
	fit!(e::Ensemble, X::Matrix, [verb])

Fit the ensemble with X.
"""
function fit!(e::Ensemble, X, verb = false)
	verb? (p = Progress(e.N, 0.1)) : nothing
	n = 0
	for model in e.models
		AnomalyDetection.fit!(model, X)
		n += 1
		verb? ProgressMeter.next!(p; showvalues = [(:"training model", "$n/$(e.N)")]) : nothing
	end
end

"""
	anomalyscore(ensemble, X)

Compute anomaly score of X.
"""
anomalyscore(e::Ensemble, X) =
	e.agregf([AnomalyDetection.anomalyscore(model,X) for model in e.models])

"""
	getthreshold(ensemble, X)

Get the automatically computed threshold.
"""
getthreshold(e::Ensemble, X) = getthreshold(e, X, e.contamination; Beta = e.Beta)

"""
	setthreshold!(ensemble, X)

Set model threshold.
"""
function setthreshold!(e::Ensemble, X)
	e.threshold = getthreshold(e, X)
end

"""
	classify(ensemble, X)

Classify X.
"""
classify(e::Ensemble, X) = Int.(anomalyscore(e, X) .> e.threshold)

"""
	predict(ensemble, X)

Predict labels for X.
"""
predict(e::Ensemble, X) = classify(e, X)