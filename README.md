# [1] 인공지능
  - 보통 사람이 행하는 지능적인 작업을 위한 연구 활동
  
# [2] 머신러닝
  - 지도학습 : 
  ex)
  - 비지도학습 : 
  ex)
  - 강화학습 : 
  ex)
  - Shallow learning(1~2개의 layer 학습)

# [3] 딥러닝
  - 머신러닝의 한 분야인 인공 신경망(ANN) 기술의 집합
  - ANN(생체 신경망 구조와 유사) - 입력, 은닉, 출력 계층으로 구성되며 각 계층은 여러 노드로 구성됨
  - 높은 성능을 위한 많은 데이터 수집, 충분히 큰 신경망(많은 은닉 유닛) 훈련 가능
  - 신경망 속도 향상을 위한 알고리즘(시그모이드, ReLU 등) 


# 이미지 파일 나누는 방법
1. keras dataset load(Xtrain,Ytrain,Xtest,Ytest)   
-> MNIST_FASHION_02.ipynb   
2. .csv파일(train/test, 픽셀값이 1row마다 있는 경우)   
-> DACON_MNIST_01.ipynb   
3. .zip파일(카테고리,img.jpg ...)   
-> CALTECH101.ipynb    
4. train안에 cat, dog 이미지만 존재할 경우(train,test,validation 파일로 분리방법, )      
-> KAGGLE_CatDog_01.ipynb   
