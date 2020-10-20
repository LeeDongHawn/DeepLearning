[UST - 파이썬딥러닝] 김남신 교수님 강의를 듣고 정리함
- Tensorflow : 2.3.0, Python : 3.7.9, Keras : 2.4.3 
- 1LinearRegression   
: x 1개로 y 예측   
- 2Multi-LinearRegression   
: x 여러개로 y 예측   
- 3Logistic Regression(binary)     
: 실 생활에서는 모든 문제의 결과를 선형(Linear)으로 표현 불가   
: Sigmoid, ReLU 등의 activation function 사용   
: Logistic_Regression_FileOpen(.csv 파일에 있는 데이터를 통해 결과 예측)
- 4Logistic Regression(multi-class)    
: 여러 개의 결과 중 1개를 선택하는 문제   
: softmax와 one-hot encoding을 통해 분류   
- 5중간고사 - 딥 러닝 모델 구축하기(binary classification)   
[딥러닝 고려사항]   
1) Preprocessing data - sklearn   
- 표준화(standardization)는 데이터를 0을 중심으로 양쪽으로 분포시킴
- 정규화(normalization)는 데이터를 특정 구간으로 나눔
- StandardSclar : 데이터를 평균이 0, 분산이 1인 값으로 변환
- MinMaxSclar : 데이터값을 0과 1사이의 범위 값으로 변환(음수는 -1~1)
- RobustScaler : StandardSclar와 빗스하지만 평균과 분산대신 median과 quartile을 사용(이상치 영향 줄임)   
2) Initializers
- 설정하지 않으면 random으로 가중치 초기화됨   
- Xavior 초기화 방법, He 초기화 방법이 있음   
- He 초기화 방법은 Xavior 초기화 방법 약간 개선 v   
3) Regularizier   
- L1규제는 가중치의 절대값에 비례하는 weight 추가   
- L2규제는 가중치의 제곱에 비례하는 weight 추가   
- L1규제는 일부 가중치 파라미터를 0으로 만듦.   
- L2 규제는 가중치 파라미터를 제한하지만 완전히 0으로 만들지 않아 더 많이 사용됨   
4) Dropout   
- 신경망에서 가장 효과적이고 널리 사용하는 규제 기법 중 하나   
- 훈련하는 동안 층의 출력 특성을 랜덤하게 0으로 만든다.   
5) Deep & Wide Neural Network 확장   
- Unit node 증가, layer 추가   
6) Activation Function   
- Sigmoid : 주로 2개의 class 분류 시 output layer에 사용 v   
- Softmax : 주로 n개의 class 분류 시 output layer에 사용   
- tanh : -1 ~ 1 사이의 값 출력, feature 값 범위 줄여주는 역할   
- ReLU : 입력 < 0 = 0, 입력 > 0 = Linear 처럼 동작, 학습 속도가 빠름 v   
7) Optimizier   
- SGD(Stochastic gradient descent) : 확률적으로 선택한 하나의 데이터로 경사 구함   
- Momentum : SGD에서 계산된 gradient에 한 스텝 전의 gradient를 일정 % 반영하여 사용(원래 gradient 유지하면서, 새로운 gradient 적용)   
- Adagrad : learning rate를 normalization   
- RMSprop : 모든 경사를 더하는 대신 지수이동평균을 사용. Non-stationary한 데이터 학습 시 주로 사용   
- Adam : Adagrad와 비슷, 0으로 편향된 것을 보정. 가장 성능이 좋다고 평가되고 있음 v   
8) loss function   
- mean square error : 오차 제곱에 대해 평균을 취함 v      
- binary classification : output layer sigmoid ( 0 or 1 )   
- categorical_crossentropy : output layer softmax ( 2 more class )   
- sparse_categorical_crossentropy : output layer softmax ( 0 or 1 )   
9) Learning Rate 조정   
- 0.01 혹은 0.001 등 적절히 선택   
10) Batch size     
- 32 or 64 등 적절히 선택   
11) Epoch   
- 적절히 선택(작동 시간 고려)   
* Batch와 Epoch 간의 관계 - 정확도에 영향을 미침
