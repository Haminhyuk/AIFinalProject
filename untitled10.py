from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
import numpy as np

cancer = load_breast_cancer()
cancer.data = cancer.data/255.0
x_train = cancer.data; x_test = cancer.data
y_train = np.int16(cancer.target); y_test = np.int16(cancer.target)

mlp = MLPClassifier(hidden_layer_sizes=(200), learning_rate_init=0.001,
                    batch_size=512, solver='adam', verbose=True)

mlp.fit(x_train, y_train) 

res = mlp.predict(x_test)

conf = np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 
    
print(conf)

correct = 0
for i in range(2): 
    correct += conf[i][i] 
accuracy = correct/len(res)
print("테스트 집합에 대한 정확률", accuracy*100, "%.")

