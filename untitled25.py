from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.model_selection import train_test_split

cancer = load_wine()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.6)

p = Perceptron(max_iter=100, eta0=0.0001) 
p.fit(x_train, y_train) 

res = p.predict(x_test) 


conf = np.zeros((3,3)) 
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 
    
print(conf)

correct = 0
for i in range(3):
    correct += conf[i][i] 
accuracy = correct/len(res)
print("Accuracy is", accuracy*100, "%.")



