#다층 퍼셉트론 프로그래밍(MNIST 데이터셋)

#fetch_openml쓰면 mnist 데이터 가져올수 있음
from sklearn.datasets import fetch_covtype
from sklearn.neural_network import MLPClassifier
import numpy as np

#mnist 데이터 불러오기 784 = 28*28 픽셀 데이터
mnist = fetch_covtype()
#0~255를 0~1 사이로 정규화
mnist.data = mnist.data/255.0
#6만개 데이터 받아오기
x_train = mnist.data[:60000]; x_test = mnist.data[60000:]
#16진수로 받아오기 -> target이 문자(string)기 때문에 숫자로 변환
y_train = np.int16(mnist.target[:60000]); y_test = np.int16(mnist.target[60000:])

mlp = MLPClassifier(hidden_layer_sizes=(100), learning_rate_init=0.001,
                    batch_size=512, solver='adam', verbose=True)

mlp.fit(x_train, y_train) #시간이 걸리는 부분

res = mlp.predict(x_test)

conf = np.zeros((10,10)) #10x10 매트릭스를 만들거야
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 #행과 열
    
print(conf)

#정확도 계산(대각선 부분만 더하기)
correct = 0
for i in range(10): #대각선 개수
    correct += conf[i][i] #대각선만 해당하는 (i,i)
accuracy = correct/len(res)
print("테스트 집합에 대한 정확률", accuracy*100, "%.")
