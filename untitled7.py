from sklearn.linear_model import Perceptron

X = [[0,0], [0,1],[1,0],[1,1]] #벡터 4개
y = [-1,1,1,1] #label

p = Perceptron() #퍼셉트론 함수 선언
p.fit(X,y) #학습

print("학습된 퍼셉트론의 매개변수: ", p.coef_, p.intercept_)
print("훈련집합에 대한 예측: ", p.predict(X))
print("정확률 측정: ", p.score(X,y)*100, "%")

