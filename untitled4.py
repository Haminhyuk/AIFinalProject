#데이터를 훈련집합 + 테스트집합으로 나누어서 학습, 예측 수행
#데이터를 60대 40으로 나뉘어서 60은 훈련, 40은 테스트 할때사용

from sklearn import datasets
from sklearn import svm

#데이터를 쪼개주는 함수 선언
from sklearn.model_selection import train_test_split
#매트릭스를 만들기 위해서 사용
import numpy as np


digit = datasets.load_digits() #데이터 불러오기

#x=feature, y= label
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)


s=svm.SVC(gamma=0.001) #분류해주는 빈 함수 모델 생성
s.fit(x_train, y_train) #fit이 s를 학습(모델링) (features, label)

res = s.predict(x_test) #new feature를 넣어서 예측을 해라

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