#다층 퍼셉트론 프로그래밍

from sklearn import datasets
#멀티레이어 퍼셉트론 MLP 분류기 임포트(MPL모델 객체 생성)
from sklearn.neural_network import MLPClassifier
import numpy as np
#데이터를 쪼개주는(테스트와 트레인) 함수 선언
from sklearn.model_selection import train_test_split
#매트릭스를 만들기 위해서 사용
import numpy as np


digit = datasets.load_digits() #데이터 불러오기
#x=feature, y= label
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)

#MLP 빈 모델 생성
#(은닉층 노드 개수, 첫번째 러닝레잇, 전체 훈련데이터(32), 솔버명시, 학습과정 온오프)
mlp = MLPClassifier(hidden_layer_sizes=(100),
                    learning_rate_init=0.001,
                    batch_size=32,
                    solver='sgd',
                    verbose=True)

mlp.fit(x_train, y_train)

res = mlp.predict(x_test) #new feature를 넣어서 예측을 해라


conf = np.zeros((10,10)) #10x10 매트릭스를 만들거야
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 #행과 열
    
print(conf)

#정확도 계산(대각선 부분만 더하기)
correct = 0
for i in range(10): #대각선 개수
    correct += conf[i][i] #대각선만 해당하는 (i,i)
accuracy = correct/len(res)
print("Accuracy is", accuracy*100, "%.")

#svm보다는 떨어지지만 일반적인 퍼셉트론보다 정확도가 매우 높다.