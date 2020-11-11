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
  모델이란? 데이터들의 패턴을 대표할 수 있는 함수   
  residual(잔차) = 실제 관측값 - 표본집단의 회귀식에서 예측된 값  
  error(오차) = 실제 관측값 - 모집단의 회귀식에서 예측된 값   
  최소자승법(Least Square Method) : 모델의 파라미터를 구하기 위한 방법, residual의 제곱을 최소화   
  linear : 미지수의 최고차항의 차수가 1을 넘지 않는 다항 방정식    
  [최적화기법] - non linear
  cost,loss,error을 최소화하고, score,이윤을 최대화하는 파라미터를 찾는 문제   
  Gradient란? 다변수 함수 f를 각 변수로 편미분한 값으로 구성되는 벡터 f(x,y)=x^2+y^2 -> gradient=(2x,2y)      
  gradient descent 방법은 steepest descent 방법이라고도 불리는데,   
  함수 값이 낮아지는 방향으로 독립 변수 값을 변형시켜가면서 최종적으로는 최소 함수 값을 갖도록 하는 독립 변수 값을 찾는 방법이다.    
  (앞이 보이지 않는 안개가 낀 산을 내려올 때 모든 방향으로 산을 더듬어가며 산의 높이가 가장 낮아지는 방향으로 진행한다.)   
  (일차미분을 통한 최적화, 이차미분을 통한 최적화)    
  [최적화기법] - linear(목적 함수가 ∑ 에러^2 의 형태인 경우)      
  SGD(Stochastic Gradient Descent), Momentum, Adagrad 등(-> optimizer)   
  ```
  - Shallow learning(입력과 출력층만 존재)

# [3] 딥러닝
``` 
  - 머신러닝의 한 분야인 인공 신경망(Artificial Neural Network) 기술의 집합
  - ANN(생체 신경망 구조와 유사) - 입력, 은닉, 출력 계층으로 구성되며 각 계층은 여러 노드로 구성됨
  - 효율적인 하드웨어 가용성(GPU)   
  - 대용량 데이터 소스와 저렴한 저장소의 이용 가능   
  - 신경망을 훈련하는데 사용하는 최적화 알고리즘의 발전   
    주로 SGD(Stochastic Gradient Descent) 사용하였으나, local minima와 느린 수렴속도 문제가 있음   
  - 모델의 정확도가 제일 중요함(성능은 환경에 따라 변화할 수 있음)   
  - 텐서플로(Tensorflow) 
  저수준 api(weight와 같은 값들을 직접 선언 가능)   
  static graph : 일단 graph(모델)을 구축하면, 동일한 graph를 계속 사용함   
  코드 순서를 변경하는 등의 방법으로 graph 개선 가능   
  - 케라스(Keras)   
  장점 : 텐서플로 같은 하위 수준 딥러닝 프레임워크(백엔드)를 기반으로 작동한다. 사용하기 쉬운 고수준 API 제공    
  단점 : 오류 발생 시 문제 해결의 어려움 존재   
  - 파이토치(Pytorch)   
   dynamic graph : 반복할 때마다 새로운 조건으로 graph 수정 가능함    
  - 테아노(Theano) : 최초의 딥러닝 프레임워크, 현재 개발 및 지원 중단   
  - Why GPU? 좋은 이유가 뭔지?      
  GPU는 여러 명령을 동시처리하는 SIMT(Single Instruction Multiple Data)구조로 되어있다.   
  GPU는 많은 행렬 곱셈을 병렬화하여 동시에 연산을 수행할 수 있도록 한다.   
  + 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?    
  GPU는 CPU로부터 작업을 할당받는데, CPU가 GPU 처리속도를 따라가지 못하는 경우가 발생함    
  + GPU를 두개 다 쓰고 싶다. 방법은? 필요한 작업을 분산한다.     
  초매개변수(hyperparameter) 병렬성 : 신경망의 서로 다른 매개변수들을 서로 다른 프로세서에서 훈련한다.   
  모형(model) 병렬성 : 모델을 여러 부분으로 나누어 각각 서로 다른 GPU에 배정하는 기법을 말합니다.      
  https://kakaobrain.com/blog/66   
  자료(data) 병렬성 : DataParallel은 당신의 데이터를 자동으로 분할하고 여러 GPU에 있는 다수의 모델에 작업을 지시합니다.   
  https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html      
 -> 나머지 내용들은 '파이썬딥러닝' 참조    
  ```

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
