from sklearn import datasets
from sklearn import svm

digit = datasets.load_digits() #데이터 불러오기

s=svm.SVC(gamma=0.1, C=10) #분류해주는 빈 함수 모델 생성
s.fit(digit.data, digit.target) #fit이 s를 학습(모델링) (features, label)

#훈련 데이터 3개만 선별(특징만 가져옴)
new_d = [digit.data[0], digit.data[1], digit.data[2]]
results = s.predict(new_d) #예측

print("예측값:", results)
print("참값: ", digit.target[0], digit.target[1], digit.target[2])


results_2 = s.predict(digit.data) #모든 피쳐를 다 가지고 예측을 해라

correct = [i for i in range(len(results_2)) if results_2[i] == digit.target[i]]
accuracy = len(correct)/len(results_2) #정확도
print("정확도: ", accuracy*100, "%")


