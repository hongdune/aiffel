```python
!pip install pillow   

from PIL import Image
import os, glob

print("PIL 라이브러리 import 완료!")
```

    Requirement already satisfied: pillow in ./anaconda3/lib/python3.7/site-packages (7.0.0)
    PIL 라이브러리 import 완료!



```python
import numpy as np

def load_data(img_path):
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data=2165   # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1       
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는",idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
```

    학습데이터(x_train)의 이미지 개수는 2165 입니다.
    x_train shape: (2165, 28, 28, 3)
    y_train shape: (2165,)



```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# model을 직접 만들어 보세요.
# Hint! model의 입력/출력부에 특히 유의해 주세요. 가위바위보 데이터셋은 MNIST 데이터셋과 어떤 점이 달라졌나요?
# [[YOUR CODE]]
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 16)        448       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 16)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 32)        4640      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 800)               0         
    _________________________________________________________________
    dense (Dense)                (None, 32)                25632     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                330       
    =================================================================
    Total params: 31,050
    Trainable params: 31,050
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)
```

    Epoch 1/15
    68/68 [==============================] - 0s 1ms/step - loss: 0.0367 - accuracy: 0.9912
    Epoch 2/15
    68/68 [==============================] - 0s 1ms/step - loss: 0.0364 - accuracy: 0.9894
    Epoch 3/15
    68/68 [==============================] - 0s 1ms/step - loss: 0.0354 - accuracy: 0.9880
    Epoch 4/15
    68/68 [==============================] - 0s 1ms/step - loss: 0.0082 - accuracy: 0.9977
    Epoch 5/15
    68/68 [==============================] - 0s 1ms/step - loss: 9.7128e-04 - accuracy: 1.0000
    Epoch 6/15
    68/68 [==============================] - 0s 1ms/step - loss: 7.1751e-04 - accuracy: 1.0000
    Epoch 7/15
    68/68 [==============================] - 0s 1ms/step - loss: 5.2262e-04 - accuracy: 1.0000
    Epoch 8/15
    68/68 [==============================] - 0s 1ms/step - loss: 3.8474e-04 - accuracy: 1.0000
    Epoch 9/15
    68/68 [==============================] - 0s 1ms/step - loss: 3.3756e-04 - accuracy: 1.0000
    Epoch 10/15
    68/68 [==============================] - 0s 1ms/step - loss: 2.8517e-04 - accuracy: 1.0000
    Epoch 11/15
    68/68 [==============================] - 0s 1ms/step - loss: 2.3930e-04 - accuracy: 1.0000
    Epoch 12/15
    68/68 [==============================] - 0s 1ms/step - loss: 2.1815e-04 - accuracy: 1.0000
    Epoch 13/15
    68/68 [==============================] - 0s 1ms/step - loss: 2.0241e-04 - accuracy: 1.0000
    Epoch 14/15
    68/68 [==============================] - 0s 1ms/step - loss: 1.8438e-04 - accuracy: 1.0000
    Epoch 15/15
    68/68 [==============================] - 0s 1ms/step - loss: 1.6326e-04 - accuracy: 1.0000





    <tensorflow.python.keras.callbacks.History at 0x7fb70d0496d0>




```python
def load_data(img_path):
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data=300   # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1       
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_test)의 이미지 개수는",idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/Downloads/rock_scissor_paper"
(x_test, y_test)=load_data(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))
```

    학습데이터(x_test)의 이미지 개수는 300 입니다.
    x_test shape: (300, 28, 28, 3)
    y_test shape: (300,)



```python
test_loss, test_accuracy = model.evaluate(x_test,y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```

    10/10 - 0s - loss: 4.8110 - accuracy: 0.6233
    test_loss: 4.810976028442383 
    test_accuracy: 0.6233333349227905


훈련 반복 횟수에 따라 모델의 정확도가 달라질 뿐만 아니라 같은 횟수여도 결과값이 다르다는 것에 해당 하이퍼 파라미터의 설정에 대한 확신이 없어져 최적화된 하이퍼 파라미터의 값을 구하는 다른 방법이 필요하다는 것을 느꼈다.
또한 처음에는 트레이닝 데이터셋의 수를 300으로 한정짓자 더 나은 예측 모델을 위해 더 질 좋은 데이터를 선별하고자 헀으나 그보단 데이터셋의 수를 늘리는 편이 훨씬 낫다는 것을 알 수 있었다.
