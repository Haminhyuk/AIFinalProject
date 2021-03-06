from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

wine = load_wine()

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, train_size=0.6)

mlp = MLPClassifier(hidden_layer_sizes=(100), learning_rate_init=0.0001,
                    batch_size=32, solver='adam', verbose=True)

mlp.fit(x_train, y_train) 

res = mlp.predict(x_test)

conf = np.zeros((3,3))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 
    
print(conf)

correct = 0
for i in range(3): 
    correct += conf[i][i] 
accuracy = correct/len(res)
print("테스트 집합에 대한 정확률", accuracy*100, "%.")


