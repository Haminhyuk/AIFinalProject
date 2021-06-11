#퍼셉트론 프로그래밍

from sklearn import datasets
from sklearn.linear_model import Perceptron
import numpy as np
#데이터를 쪼개주는 함수 선언
from sklearn.model_selection import train_test_split
#매트릭스를 만들기 위해서 사용
import numpy as np


digit = datasets.load_digits() #데이터 불러오기
#x=feature, y= label
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)

p = Perceptron(max_iter=100, eta0=0.001) # epoch(1세대,100번 반복학습), learning rate
p.fit(x_train, y_train) #fit이 p가 학습(모델링) (features, label)

res = p.predict(x_test) #new feature를 넣어서 예측을 해라


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