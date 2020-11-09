# [1] 인공지능
  - 보통 사람이 행하는 지능적인 작업(인지,추론)을 기계가 수행하도록 하기 위한 연구 활동   
  - 약 인공지능 : 특정 문제 해결(인간 두뇌의 특정 일부 기능 모사)           
  - 강 인공지능 : 사람과 같은 사고가 가능(인간을 대체할 수준, 이성적 판단)         
  
# [2] 머신러닝(인공지능을 위한 방법)   
  ```   
  - 지도학습 : 정답을 알려주며 학습시키는 것(input data, labeling)     
    Regression(회귀) : 어떤 데이터들의 특징을 통해 값 예측(동네 평수 가격)      
      Linear(Multi) Regression - x와 y(1:1, n:1) : x를 통해 y를 예측할 수 있는 경우  
      Logistic Regression - 모든 문제를 직선 형태로 표시할 수 없다.(0과 1사이 범위)   
      Sigmoid, ReLu, Softmax(출력 값 범위 지정, activation function)   
    Classification(분류) : 이진 분류(0 or 1), 다중 분류(개,고양이,토끼..)    
      서포트 벡터 머신(SVM) : 다른 class에 속하는 데이터 그룹 사이에 결정 경계를 찾는다.
      k-nn : 특정 데이터를 선택하고, 특정 데이터에서 가까운 k개를 통해 특정 데이터의 class를 결정한다.   
      의사결정 트리 : 입력 데이터를 조건문을 통해 계속 분류하여 출력값을 예측한다.
      랜덤 포레스트 : 입력 데이터들을 섞어서 여러 개의 의사결정 트리를 만들고, 그 중에서 가장 좋은 결과를 선택한다.   
      인공신경망(ANN)    
  ```   
  ```   
  - 비지도학습 : 정답을 따로 알려주지 않고, 비슷한 데이터들을 군집화 하는 것   
  * Clustering(K-Means) : 주어진 데이터 중 유사한 데이터 그룹을 찾는 것     
  ```    
  ```   
  - 강화학습 : 상과 벌이라는 보상을 통해 상을 최대화하고 벌을 최소화 하도록 학습하는 방식   
  ```     
  ```   
  [WorkFlow]   
  데이터 검색 : 데이터 수집과 추출(공공 데이터셋, 개인 데이터셋 사용가능)      
  데이터 준비(오래걸림) : 데이터를 사용 가능한 형태로 변환, 특성 추출(너무 많으면 처리 어려움 - 차원의 저주) 등 수행     
  PCA(Principal Component Analysis) : 데이터의 feature간의 상관관계를 통해 주성분(분산이 가장 큰)을 추출해 차원을 축소    
  모델링 : 데이터를 입력하고, 여러 방법들을 사용하여 모델 훈련     
  모델 평가와 튜닝 : 모델링, 평가, 튜닝을 통해 최상의 성능을 내는 모델 탐색
  (혼동행렬-예측과실제값간의 TP(참참), FP(거짓인데참), TN(참인데거짓), FN(거짓거짓)) 등 지표로 사용        
  배포(모니터링)   
  ```
  ```   
  [머신러닝 - 모델 선정]   
  
  ```
  - Shallow learning(입력과 출력층만 존재)

# [3] 딥러닝
  - 머신러닝의 한 분야인 인공 신경망(ANN) 기술의 집합
  - ANN(생체 신경망 구조와 유사) - 입력, 은닉, 출력 계층으로 구성되며 각 계층은 여러 노드로 구성됨
  - 높은 성능을 위한 많은 데이터 수집, 충분히 큰 신경망(많은 은닉 유닛) 훈련 가능
  - 신경망 속도 향상을 위한 알고리즘(시그모이드, ReLU 등) 


# 이미지 파일 나누는 방법
1. keras dataset load(Xtrain,Ytrain,Xtest,Ytest)   
-> MNIST_FASHION_02.ipynb   
2. .csv파일(train/test, 픽셀값이 1ROW마다 존재하는 경우)   
-> DACON_MNIST_01.ipynb   
3. .zip파일(|카테고리,img.jpg| 반복 ...)   
-> CALTECH101.ipynb    
4. train안에 cat, dog 이미지만 존재할 경우(train,test,validation 파일로 이미지 분리, 제출용아님)      
-> KAGGLE_CatDog_01.ipynb   
5. (4번과 동일 데이터셋) 폴더 안 이미지 이름에 따라 카테고리 분류(이미지명.jpg, 카테고리) 후       
pandas dataframe 형태로 만듦 dataframe으로부터 train, validation 분리한 다음,      
test 데이터 검증 수행(전체 데이터 활용하기 때문에 시간 굉장히 오래걸림, 방법만 참조)      
-> KAGGLE_CatDog_02.ipynb    
6. Pre-Trained Model   
따로 학습과정 없이, 예측(IMAGENET에 있는 class)하고자 하는 이미지(2개)만 읽어서 바로 예측 실행   
-> KAGGLE_CatDog_PreTrained.ipynb   
7. Pre-Trained Model + Transfer Learning  
(IMAGENET에 속하지 않는 이미지 class에 대하여) VGG16모델 구조를 FC layer만 변경, yes or no 판별   
데이터셋 구조 : train(절/신사), validation(절/신사), test(절+신사 섞임)   
-> Transfer_Learning.ipynb, utils.py   
