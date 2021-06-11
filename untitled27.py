from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

wine = load_wine() 

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, train_size=0.6)

s=svm.SVC(gamma=0.00000001)
s.fit(x_train, y_train)

res = s.predict(x_test) 

conf = np.zeros((3,3)) 
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 
    
print(conf)

correct = 0
for i in range(3): 
    correct += conf[i][i] 
accuracy = correct/len(res)
print("Accuracy is", accuracy*100, "%.")

