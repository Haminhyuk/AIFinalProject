from sklearn import datasets

d=datasets.load_iris() #iris 데이터셋을 읽고(laod_iris 함수 호출)
print(d.DESCR)         #내용을 출력(DESCR=변수)

#iris 데이터 내용 조회
for i in range(0, len(d.data)): # 샘플을 순서대로 출력
    print(i+1,d.data[i],d.target[i])
    
from sklearn import svm

s=svm.SVC(gamma=0.1, C=10)
s.fit(d.data, d.target)

new_d=[[6.4,3.2,6.0,2.5],[7.1,3.1,4.7,1.35]]

res=s.predict(new_d) #예측하다
print("새로운 2개 샘플의 부류는", res)