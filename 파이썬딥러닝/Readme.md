[UST - 파이썬딥러닝] 김남신 교수님 강의를 듣고 정리함(keras)   
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
>>> - strong : 0.01, week : 0.0001   
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
>>> - mean_square_error : 오차 제곱에 대해 평균을 취함         
>>> - binary_crossentropy  : output layer sigmoid ( 0 or 1 )   
>>> - categorical_crossentropy : output layer softmax ( 2 more class )   
>>> - sparse_categorical_crossentropy : output layer softmax ( 0 or 1 )   
>> 9) Learning Rate 조정   
>>> - 0.01 혹은 0.001 등 적절히 선택    
>> 10) Batch와 Epoch와 Iteration
>>> - Epoch : 인공 신경망에서 전체 데이터셋에 대해 forward/backward 학습을 완료한 횟수   
>>> - Batch : 메모리 한계와 속도 저하 때문에 모든 데이터를 한번에 집어넣을 수 없어, 데이터를 나누는 것(데이터 size)   
>>> - Iteration : 몇 번 나누는지에 대한 횟수
>>> ex) 2000개 Train data에 대해 epochs = 20, batch_size = 500인 경우   
>>> 1epoch은 (2000/500) 4회 Iteration으로 학습이 진행된다.   
>>> 20epoch결과 전체 20번의 학습이 이루어지며, (2000/500)*20 80회 Iteration으로 학습이 진행된다.     
    
- 6MNIST - MNIST 숫자 분류하기, Fashion_MNIST 분류하기(시각화 기법 추가)         
> fit() : 모델 훈련(x_train,y_train + x_valid,y_valid), predict() : 준비된 test셋에 대한 출력값(x_test,y_test), evaluate(x_test,y_test) : 임의의 입력을 통해 예측       
>> 1) MNIST_1~5 : Using DNN(Not CNN) 
>>> 1 : Only 1 layer, 2 : More than 2 layer, 3 : Initializer 설정, 4 : Deep and Wide, 5 : Dropout   
>> 2) MNIST_6 : Using CNN(따로 기법 적용 x, 아래 구조로만 구성됨)   
>> 3) MNIST_LeNet : Using CNN - LeNet(따로 기법 적용 x, LeNet구조 사용)   
>> 4) MNIST_LeNet2_GAP : Flatten() 대신 GlobalAveragePooling 사용   
```
CNN Model   
1) 구조   
- 1[Conv + ReLu] * N(0<=N<=3)   
+ BatchNormalization() : 학습하는 과정 안정화하여 학습 속도 가속     
학습의 불안정화 - Internal Covariate Shift   
레이어를 통과할 때마다 현재 레이어의 입력 분포가 변하는 현상     
배치 정규화는 간단히 말하자면 미니배치의 평균과 분산을 이용해서 정규화 한 뒤에,   
scale 및 shift 를 감마(γ) 값, 베타(β) 값을 통해 실행한다. 이 때 감마와 베타 값은 학습 가능한 변수이다.   
즉, Backpropagation을 통해서 학습이 된다.     
컨볼루션 레이어에서 활성화 함수가 입력되기 전에 WX + b 로 가중치가 적용되었을 때,     
b의 역할을 베타가 완벽히 대신 할 수 있기 때문에 b 를 삭제한다. 또한 CNN의 경우 컨볼루션 성질을 유지 시키고 싶기 때문에 각 채널을 기준으로 각각의 감마와 베타를 만들게 된다.    
예를 들어 미니배치가 m 채널 사이즈가 n 인 컨볼루션 레이어에서 배치 정규화를 적용하면 컨볼루션을 적용한 후의 특징 맵의 사이즈가 p x q 일 경우,    
각 채널에 대해 m x p x q 개의 스칼라 값(즉, n x m x p x q 개의 스칼라 값)에 대해 평균과 분산을 구한다.    
최종적으로 감마 베타 값은 각 채널에 대해 한 개씩, 총 n개의 독립적인 배치 정규화 변수 쌍이 생기게 된다.    
즉, 컨볼루션 커널 하나는 같은 파라미터 감마, 베타를 공유하게 된다.   
출처: https://eehoeskrap.tistory.com/430 [Enough is not enough]
-> Dropout 대체하는 기법임   
- 2[Pooling] * M(M>=0)   
- (1 2), (1 2) 여러 번 반복 수행 가능   
+ Flatten() or GlobalAveragePooling() 
- 3[FC + ReLu] * K(0<=K<=2)   
(마지막 : Softmax) 
   
2) layer  
- Conv2D layer    
용도 : 이미지에서 특징을 추출(Activation Map)하기 위함   
: input_shape = (height, width, channel), activation = 'None', kernel_initializer="glorot_uniform", data_format=None(input_shape 순서 설정)      
: filters = 개수, 입력 데이터를 지정된 간격(stride)로 순회하며 채널 별로 합성곱을 수행하고 모든 채널(ex, RGB 3개)의 합성곱 결과를 더하여 Feature(=Activation) Map을 생성한다.     
(input 이미지 : MxM, no padding)필터의 개수 = Activation maps의 개수, output_size=1+(M-kernel_size)/stride, Activation map = (size,size,필터개수)    
여러 개의 작은 크기의 필터를 사용하는 것이 좋음   
: stride=(1,1), stride : 지정한 간격으로 필터를 움직이며 합성곱을 수행   
: padding="valid/same", padding : 입력과 동일한 높이와 너비를 가진 특징 맵을 얻기위한 방법(외각에 0으로 데이터 채움)   
zero padding 사용하는 것이 영상 크기 유지와 경계면 정보를 유지할 수 있어 더 좋다.   
: ReLU - 속도와 정확도 면에서 성능이 뛰어남  
-> 파라미터 : (input_channel x filter_size(k*k) x output_channel(=filter개수)) + output bias(=filter 개수)     
처음 이후 input_channel은 이전 conv의 filter 개수     
   
- (Max/Average/GlobalMax/GlobalAverage)Pooling2D layer   
용도 : Activation Map 크기를 줄이거나 특정 데이터 강조하기 위함, pool_size만큼 stride간격으로 순회   
Pooling 연산은 Activation Map의 개수(filter 개수)를 줄이지 않는다.(크기만 줄여줌)       
: MaxPooling = pool_size에 있는 가장 큰 값으로 새로운 Activation Map을 생성   
: AveragePooling = pool_size에 있는 값의 평균으로 새로운 Activation Map을 생성   
: GlobalMaxPooling = Activation Map 전체에서 가장 큰 값 1개만 추출함   
: Pooling다음 Drop_out 놓기도 함   
   
- Fully Connected layer    
용도 : 추출된 특징 값을 Neural Network에 넣어서 최종 분류까지 수행   
: Flatten() or GlobalAveragePooling()   
-> Flatten()은 파라미터 개수 증가, 계산오래걸림, 데이터의 Shape만 변경해주는 계층    
-> GAP()는 파라미터 개수 감소, 계산시간단축   
: ReLu + Drop_out + Softmax 등으로 구성(3개 이상 사용하지 않는 것이 좋음)      
   
3) 유명한 CNN Model   
[ImageNet] 학습한 모델, 이 모델을 그대로 불러와 사용하는 것을 사전학습모델(Pre-Trained Model)을 사용한다고 한다.   
-> 앞쪽 층의 필터들은 윤곽선같은 저수준 특징들을 담은 특징 맵을 산출하고,   
뒤쪽 층의 필터들은 좀 더 복잡한 ex)눈, 귀 특징을 생성한다.   
ImageNet에 존재하지 않는 사진들을 분류하고 싶을 때는 어떻게 해야할까?   
사전학습 모델을 약간 수정하여 사용한다.(처음부터 가중치 학습하는 것보다 정확도, 속도 향상 가능) 이를 전이학습이라고 한다.    
(ex, Fully connected layer 수정)   
- LeNet-5(1998), AlexNet(2012), ZFNet(2013), VGG(2014), GoogLeNet(=Inception, 2014), ResNet(2015)      
```    
   
```    
CNN 관련 기법들(업데이트 예정)      
1) Train, Test, Validation split   
용도 : Train|Test 로만 나누면 overfitting 발생 가능 -> Train|Validation|Test로 데이터 분리   
-> 사용법 : MNIST_Fashion02 : (x_train, y_train, x_test, y_test)로 나눠진 데이터셋(load_dataset으로 불러옴)   
-> 사용법 : DACON_MNIST : .csv 파일(train/test), 이미지마다 픽셀이 1 row에 나열되어 있음   

2) K-fold(sklearn의 StratifiedKFold)            
용도 : overfitting 방지   
overfitting : 학습 데이터셋은 정확도 높으나, 새로운 데이터 적용 시 잘 맞지 않는 것   
: 현재 상태 - Train|Test   
: k지정(ex, k=3) Train = Train(2/3)|Validation(1/3)으로 나뉨(ㅁ : Train, O : Validation)     
: 1회차 - ㅁ ㅁ O 로 나눠 학습과 성능 측정   
: 2회차 - ㅁ O ㅁ 로 나눠 학습과 성능 측정   
: 3회차 - O ㅁ ㅁ 로 나눠 학습과 성능 측정   
: 1,2,3회차 결과에 대한 평균 Hyperparameters을 최종적으로 Train Data에 적용   
: 마지막으로 Test Data에 대해 평가   
-> 사용법 : DACON_MNIST_03, CALTECH101_Upgrade

3) ImageDataGenerator  
용도 :  데이터 증강기법     
idg_train = ImageDataGenerator() 
rescale = 1/255. (입력값 0~1 변환)   
3-1) flow_from_directory   
용도 : 데이터 읽기 위한 iterator   
img_itr_train = idg_train.flow_from_directory()    
학습 : history = model.fit_generator()     
-> 사용법 : Transfer_Learning.ipynb   

4) ReduceLROnPlateau(callbacks)      
용도 : loss가 향상되지 않을 때 learning rate를 조정함(monitor, factor, patience, verbose)      
patience를 보고, monitor 성능이 향상되지 않으면 factor를 통해 learning rate를 변화시킨다.
-> 사용법 : MNIST_Fashion_02, DACON_MNIST_1  

5) EarlyStopping(callbacks)         
용도 : Epoch을 많이 돌리며, 특정 시점에서 멈추기 위해 사용(monitor, patience, verbose)           
verbose=1 : 언제 멈췄는지 알 수 있음, monitor='val_loss' 어느 성능이, patience : 몇 번동안 향상되지 않은 경우 종료    
-> 사용법 : MNIST_Fashion_02, DACON_MNIST_1   
   
6) ModelCheckpoint(callbacks)          
용도 : 가장 높은 검증 정확도의 모델 저장하기위해 사용   
save_best_only : 이전보다 향상된 모델 가중치 저장(가중치 load해서 test에 적용)      
-> 사용법 : MNIST_Fashion_02, DACON_MNIST_1     

7) Ensemble(정확한지 모르겠음)   
모델1(vgg16) -> 학습 -> 예측   
모델2(google) -> 학습 -> 예측   
모델3(resnet) -> 학습 -> 예측  
모델1, 모델2, 모델3의 예측 결과 중에서 최빈값을 최종 예측으로 사용   
```
   
- 7 - PyImageSearchDeepLearning      
7.1) CALTECH101, Homework_CALTECH101 : classification_report, confusion_matrix 사용법    
7.2) CALTECH101_upgrade : K-fold 사용법   
7.3) Homework_BostonHouse : mlp와 cnn 모델 합쳐 사용하는 방법      
   
- 8 - 전이학습   
8.1) T_pretrained-weight, T_imagenet-vgg-res-inception-xception-keras(모델 그대로 사용)       
8.2) Transfer_Learning, T_tutorial_transfer_learning, T_mlp_mnist(모델 수정해서 사용)    
