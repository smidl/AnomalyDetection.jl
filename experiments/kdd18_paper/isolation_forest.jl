using ScikitLearn: @sk_import
@sk_import ensemble: IsolationForest

# isoforest specific functions - sklearn uses row instances
fit!(m::PyCall.PyObject, X) = ScikitLearn.fit!(m, X')
predict!(m::PyCall.PyObject, X) = ScikitLearn.predict(m, X')
anomalyscore(m::PyCall.PyObject, X) = 1.0-ScikitLearn.decision_function(m, X')

