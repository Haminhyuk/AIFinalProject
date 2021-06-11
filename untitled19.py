from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups

digit = datasets.fetch_20newsgroups()

s = svm.SVC(gamma = 0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv = 5) 

print(accuracies)
print("평균정확률: ", accuracies.mean()*100, "%")

