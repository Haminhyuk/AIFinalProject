from sklearn import datasets
from sklearn import svm

d=datasets.load_wine()

s=svm.SVC(gamma=0.1,C=10)
s.fit(d.data,d.target)

new_d=[[1.416e+01, 2.510e+00 ,2.440e+00 ,2.000e+01 ,9.100e+01, 1.680e+00 ,7.000e-01
 ,4.400e-01, 1.240e+00, 9.740e+00 ,6.200e-01 ,1.710e+00, 6.610e+02],[1.216e+01, 1.610e+00 ,2.310e+00, 2.280e+01, 9.000e+01, 1.780e+00, 1.690e+00
 ,4.300e-01, 1.560e+00, 2.450e+00 ,1.330e+00, 2.260e+00 ,4.920e+02]]

res=s.predict(new_d)
print("새로운 샘플의 부류는", res)