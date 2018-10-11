using AnomalyDetection

name = "yeast"
alpha= 0.8
diff = ["easy", "medium"]

a,b = AnomalyDetection.getdata(name, alpha, diff)
println(size(a.data))
println(size(b.data))