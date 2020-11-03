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
> [딥러닝 Deep Neural Network 고려사항]   
>> 1) Preprocessing data - sklearn   
>>> - 표준화(standardization)는 데이터를 0을 중심으로 양쪽으로 분포시킴
>>> - 정규화(normalization)는 데이터를 특정 구간으로 나눔
>>> - StandardSclar : 데이터를 평균이 0, 분산이 1인 값으로 변환
>>> - MinMaxSclar : 데이터값을 0과 1사이의 범위 값으로 변환(음수는 -1~1)
>>> - RobustScaler : StandardSclar와 빗스하지만 평균과 분산대신 median과 quartile을 사용(이상치 영향 줄임)   
>> 2) Initializers
>>> - 설정하지 않으면 random으로 가중치 초기화됨   
>>> - Xavior 초기화 방법, He 초기화 방법이 있음   
>>> - He 초기화 방법은 Xavior 초기화 방법 약간 개선 v   
>> 3) Regularizier   
>>> - L1규제는 가중치의 절대값에 비례하는 weight 추가   
>>> - L2규제는 가중치의 제곱에 비례하는 weight 추가   
>>> - L1규제는 일부 가중치 파라미터를 0으로 만듦.   
>>> - L2 규제는 가중치 파라미터를 제한하지만 완전히 0으로 만들지 않아 더 많이 사용됨   
>> 4) Dropout   
>>> - 신경망에서 가장 효과적이고 널리 사용하는 규제 기법 중 하나   
>>> - 훈련하는 동안 층의 출력 특성을 랜덤하게 0으로 만든다.   
>> 5) Deep & Wide Neural Network 확장   
>>> - Unit node 증가, layer 추가   
>> 6) Activation Function   
>>> - Sigmoid : 주로 2개의 class 분류 시 output layer에 사용 v   
>>> - Softmax : 주로 n개의 class 분류 시 output layer에 사용   
>>> - tanh : -1 ~ 1 사이의 값 출력, feature 값 범위 줄여주는 역할   
>>> - ReLU : 입력 < 0 = 0, 입력 > 0 = Linear 처럼 동작, 학습 속도가 빠름 v   
>> 7) Optimizier   
>>> - SGD(Stochastic gradient descent) : 확률적으로 선택한 하나의 데이터로 경사 구함   
>>> - Momentum : SGD에서 계산된 gradient에 한 스텝 전의 gradient를 일정 % 반영하여 사용(원래 gradient 유지하면서, 새로운 gradient 적용)   
>>> - Adagrad : learning rate를 normalization   
>>> - RMSprop : 모든 경사를 더하는 대신 지수이동평균을 사용. Non-stationary한 데이터 학습 시 주로 사용   
>>> - Adam : Adagrad와 비슷, 0으로 편향된 것을 보정. 가장 성능이 좋다고 평가되고 있음 v   
>> 8) loss function   
>>> - mean square error : 오차 제곱에 대해 평균을 취함 v      
>>> - binary classification : output layer sigmoid ( 0 or 1 )   
>>> - categorical_crossentropy : output layer softmax ( 2 more class )   
>>> - sparse_categorical_crossentropy : output layer softmax ( 0 or 1 )   
>> 9) Learning Rate 조정   
>>> - 0.01 혹은 0.001 등 적절히 선택   
>> 10) Batch size     
>>> - 32 or 64 등 적절히 선택   
>> 11) Epoch   
>>> - 적절히 선택(작동 시간 고려)   
>> 12) Batch와 Epoch와 Iteration
>>> - Epoch : 인공 신경망에서 전체 데이터셋에 대해 forward/backward 학습을 완료한 횟수   
>>> - Batch : 메모리 한계와 속도 저하 때문에 모든 데이터를 한번에 집어넣을 수 없어, 데이터를 나누는 것(데이터 size)   
>>> - Iteration : 몇 번 나누는지에 대한 횟수
>>> ex) 2000개 Train data에 대해 epochs = 20, batch_size = 500인 경우   
>>> 1epoch은 (2000/500) 4회 Iteration으로 학습이 진행된다.   
>>> 20epoch결과 전체 20번의 학습이 이루어지며, (2000/500)*20 80회 Iteration으로 학습이 진행된다.     
    
- 6MNIST - MNIST 숫자 분류하기, Fashion_MNIST 분류하기(시각화 기법 추가)         
> fit() : 모델 훈련(x_train,y_train), predict() : 입력에 대한 출력값(x_test), evaluate(x_test,y_test) : 테스트 데이터를 통해 정확도 평가    
>> 1) MNIST_1~5 : Using DNN(Not CNN) 
>>> 1 : Only 1 layer, 2 : More than 2 layer, 3 : Initializer 설정, 4 : Deep and Wide, 5 : Dropout   
>> 2) MNIST_6 : Using CNN     
```
CNN Model   
1) 구조   
- 1[Conv + ReLu] * N(0<=N<=3)   
- 2[Pooling] * M(M>=0)   
- (1 2), (1 2) 여러 번 반복 수행 가능   
(Flatten): 데이터의 Shape만 변경해주는 계층      
- 3[FC + ReLu] * K(0<=K<=2)   
(마지막 : Softmax) 
   
2) layer  
- Conv2D layer    
용도 : 이미지에서 특징을 추출(Activation Map)하기 위함   
: input_shape = (height, width, channel), activation = 'None', kernel_initializer="glorot_uniform", data_format=None(input_shape 순서 설정)      
: filters = 개수, 입력 데이터를 지정된 간격(stride)로 순회하며 채널 별로 합성곱을 수행하고 모든 채널(ex, RGB 3개)의 합성곱 결과를 더하여 Feature(=Activation) Map을 생성한다.     
(MxM 이미지, no padding)필터의 개수 = Activation maps의 개수, size=1+(M-kernel_size)/stride, Activation map = (size,size,필터개수)     
: stride=(1,1), stride : 지정한 간격으로 필터를 움직이며 합성곱을 수행   
: padding="valid/same", padding : 입력과 동일한 높이와 너비를 가진 특징 맵을 얻기위한 방법(외각에 0으로 데이터 채움)   
   
- (Max/Average/GlobalMax/GlobalAverage)Pooling2D layer   
용도 : Activation Map 크기를 줄이거나 특정 데이터 강조하기 위함, pool_size만큼 stride간격으로 순회   
: MaxPooling = pool_size에 있는 가장 큰 값으로 새로운 Activation Map을 생성   
: AveragePooling = pool_size에 있는 값의 평균으로 새로운 Activation Map을 생성   
: GlobalMaxPooling = Activation Map 전체에서 가장 큰 값 1개만 추출함   
   
- Fully Connected layer    
용도 : 추출된 특징 값을 Neural Network에 넣어서 최종 분류까지 수행   
: ReLu + Drop out + Softmax 등으로 구성   
   
3) 유명한 CNN   
[ImageNet] 학습한 모델, 이 모델을 그대로 불러와 사용하는 것을 사전학습모델(Pre-Trained Model)을 사용한다고 한다.   
하지만, ImageNet에 존재하지 않는 사진들을 분류하고 싶을 때는 어떻게 해야할까?   
사전학습 모델을 약간 수정하여 사용한다.(처음부터 가중치 학습하는 것보다 정확도, 속도 향상 가능) 이를 전이학습이라고 한다.    
(ex, Fully connected layer 수정)   
- LeNet-5(1998), AlexNet(2012), ZFNet(2013), VGG(2014), GoogLeNet(=Inception, 2014), ResNet(2015)      
```    
   
```    
CNN 관련 기법들(업데이트 예정)      
1) Train, Test, Validation split   
용도 : Train|Test 로만 나누면 overfitting 발생 가능 -> Train|Validation|Test로 데이터 분리   
   
2) K-fold         
용도 : overfitting 방지   
: 현재 상태 - Train|Test   
: k지정(ex, k=3) Train = Train(2/3)|Validation(1/3)으로 나뉨(ㅁ : Train, O : Validation)     
: 1회차 - ㅁ ㅁ O 로 나눠 학습과 성능 측정   
: 2회차 - ㅁ O ㅁ 로 나눠 학습과 성능 측정   
: 3회차 - O ㅁ ㅁ 로 나눠 학습과 성능 측정   
: 1,2,3회차 결과에 대한 평균 Hyperparameters을 최종적으로 Train Data에 적용   
: 마지막으로 Test Data에 대해 평가   
   
3) ImageDataGenerator  
용도 :  데이터 증강기법    

3-1) flow_from_directory

4) ReduceLROnPlateau      
5) EarlyStopping      
6) ModelCheckpoint       
7) Ensemble      

등등 ..
[MNIST ImageDataGenerator]

1. train_datagen = ImageDataGenerator(데이터 증강 조건 및 validation 비율 설정(rescale, shear_range, zoom_range, horizontal_flip, validation_split ...))   
2. train_generator = train_datagen.flow_from_directory(데이터 증강 수행(data directory, target_size, batch_size, class_mode, subset ...))    
3. validation_generator = train_datagen.flow_from_directory(데이터 증강 수행(data directory, target_size, batch_size, class_mode, subset ...))   
4. model.fit_generator(모델 학습 수행(train_generator, steps_per_epoch, validation_data, validation_steps, epochs ...))   

[Load ImageDataGenerator]      

```
