#교차검증을 이용해 성능 측정
#정확률이 들쭉날쭉
#k를 크게하면? 신뢰도 향상, but 실행시간 증가 (주로 5, 10을 선택)

from sklearn import datasets
from sklearn import svm

#k값을 주면 그 k값에 맞춰서 데이터를 쪼개서 바로 학습까지 해주는 함수
from sklearn.model_selection import cross_val_score

digit = datasets.load_digits()
s = svm.SVC(gamma = 0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv = 10) #cv = k값

print(accuracies)
print("평균정확률: ", accuracies.mean()*100, "%")