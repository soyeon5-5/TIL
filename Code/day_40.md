# 22.03.04

## TensorFlow

```python
pip install tensorflow
```

#### 의류 분류-이진분류

1. 데이터 준비

   ```python
   import tensorflow as tf
   from tensorflow import keras
   import numpy as np
   import matplotlib.pyplot as plt
   ```

   ```python
   print(tf.__version__)
   ```

   ```python
   fashion_mnist = keras.datasets.fashion_mnist
   (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
   ```

   ```python
   print(train_images.shape)
   print(test_images.shape)
   print(train_labels)
   class_names=['T-shirt/top', 'Trouser','Pullover', 'Dress', 'Coat', 'Sandal','Shirt','Sneaker','Bag','Ankle boot']
   ```

2. 데이터 전처리

   ```python
   plt.figure(figsize=(10,10))
   plt.imshow(train_images[1])
   plt.colorbar()
   plt.grid(False)
   plt.show()
   # 원사이즈
   
   train_images = train_images / 255.0
   test_images = test_images/ 255.0
   # 크기 변환
   
   plt.figure(figsize=(10,10))
   plt.imshow(train_images[1])
   plt.colorbar()
   plt.grid(False)
   plt.show()
   ```

   ```python
   plt.figure(figsize=(10,10))
   for i in range(25):
     plt.subplot(5, 5, i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(train_images[i], cmap=plt.cm.binary)
     plt.xlabel(class_names[train_labels[i]])
   plt.show()
   ```

3. 모델 구성

   ```python
   # 층설정
   model= keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                            keras.layers.Dense(128, activation=tf.nn.relu),
                            keras.layers.Dense(10, activation=tf.nn.softmax)
   ])
   ```

   ```python
   # 컴파일
   model.compile(optimizer = 'adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
                 
   # 훈련
   model.fit(train_images, train_labels, epochs=5)
   ```

4. 정확도 평가

   ```python
   test_loss, test_acc = model.evaluate(test_images, test_labels) # 성능평가
   ```

5. 예측 만들기

   ```python
   predictions = model.predict(test_images)
   print(predictions[0])
   print(np.argmax(predictions[0]))
   print(test_label[0])
   ```

   ```python
   plt.figure(figsize=(10,10))
   for i in range(25):
     plt.subplot(5,5,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid('off')
     plt.imshow(test_images[i], cmap=plt.cm.binary)
   
     predicted_label = np.argmax(predictions[i]) # 예측값중 가장큰값을 라벨로
     true_label = test_labels[i]
   
     if predicted_label == true_label: # 예측과 원래 라벨 같으면 초록, 아니면 빨강
       color = 'green'
     else:
       color = 'red'
     plt.xlabel("{} ({})".format(class_names[predicted_label],
                                  class_names[true_label]),
                                  color=color)
   ```

   ```python
   # 이미지
   def plot_image(i, predictions_array, true_label, img):
     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
     plt.grid(False)
     plt.xticks([])
     plt.yticks([])
   
     plt.imshow(img, cmap=plt.cm.binary)
   
     predicted_label = np.argmax(predictions_array)
     if predicted_label == true_label:
       color = 'blue'
     else:
       color = 'red'
     
     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
                color = color)
   
   # 그래프
   def plot_value_array(i, predictions_array, true_label):
     predictions_array, true_label = predictions_array[i], true_label[i]
     plt.grid(False)
     plt.xticks([])
     plt.yticks([])
     thisplot = plt.bar(range(10), predictions_array, color = '#777777')
     plt.ylim([0,1])
     predicted_label = np.argmax(predictions_array)
   
     thisplot[predicted_label].set_color('red')
     thisplot[true_label].set_color('blue')
   ```

   ```python
   # 총 토탈 표시
   num_rows = 5
   num_cols = 3
   num_images = num_rows*num_cols
   plt.figure(figsize=(2*2*num_cols, 2*num_rows))
   for i in range(num_images):
     plt.subplot(num_rows, 2*num_cols, 2*i+1)
     plot_image(i, predictions, test_labels, test_images)
     plt.subplot(num_rows, 2*num_cols, 2*i+2)
     plot_value_array(i, predictions, test_labels)
   plt.show()
   ```

---

## Neural Network

#### 영화 리뷰 분류 - 이진분류

1. 데이터 준비

   ```python
   from keras.datasets import imdb
   
   (train_data, train_labels), (test-data, test_labels) = imdb.load_data(num_words=10000) # 가장 자주 나타나는 만개 단어를 사용하겠다는 의미
   ```

2. 데이터 전처리

   ```python
   # 정수 시퀀스를 이진 행렬로 인코딩
   import numpy as np
   
   def vectorize_sequences(seqss, dim=10000):
       results = np.zeros((len(seqs), dim))
       for i, seq in enumerate(seqs):
           resuts[i, seq] =1. # 특정 인덱스 위치를 1.로 만듬
       return results
   
   x_train = vectorize_sequences(train_data)
   x_test  = vectorize_sequences(test_data)
   ```

3. 모델 구성

   ```python
   from keras import models
   from keras import layers
   ```

   ```python
   model = models.Sequential()
   model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) #은닉층
   model.add(layers.Dense(16, activation='relu')) #은닉층
   model.add(layers.Dense(1, activation='sigmoid'))
   ```

   ```python
   model.compile(optimizer='rmrprop', loss='binary_crossentropy', metrics=['accuracy'])
   ## 똑같지만 다른 입력 2개
   # from tensorflow.keras import optimizers
   # model.complie(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
   
   # from keras import losses
   # from keras import metrics
   # model.compile(optimizer=optimizers.RMSprop(learing_rate=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
   ```

   ```python
   # 훈련
   x_val= x_train[:10000]
   partial_x_train = x_train[10000:]
   y_val= y_train[:10000]
   partial_y_train= y_train[10000:]
   
   history = model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val))
   ```

4. 그래프로 비교

   ```python
   # loss 비교
   import matplotlib.pyplot as plt
   
   history_dict = histroy.history
   loss= history_dict['loss']
   val_loss = history_dict['val_loss']
   
   epochs=range(1, len(loss)+1)
   
   plt.figure(figsize=(16,12))
   plt.plot(epochs, loss, 'bo', label='Training loss')
   plt.plot(epochs, val_loss, 'b', label='Validation loss')
   plt.title('Training and validation loss')
   plt.xlabel('Epochs')
   plt.ylabel('loss')
   plt.legend()
   plt.show()
   ```

   ```python
   # acc 비교
   plt.clf()
   acc = history_dict['acc']
   val_acc = history_dict['val_acc']
   
   epochs=range(1, len(loss)+1)
   
   plt.figure(figsize=(16,12))
   plt.plot(epochs, acc, 'bo', label='Training acc')
   plt.plot(epochs, val_acc, 'b', label='Validation acc')
   plt.title('Training and validation acc')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()
   ```

5. Epoch 조절

   ```python
   # 과적합 방지
   # 다시 모델 만들고 훈련
   model = models.Sequential()
   model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
   model.add(layers.Dense(16, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
   
   model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=4, batch_size=512)
   results = model.evaluate(x_test, y_test)
   print(results)
   ```

6. 예측

   ```python
   model.predict(x_test)
   ```

7. 추가 실험

   ```python
   # 은닉층 1개
   model = models.Sequential()
   model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
   model.add(layers.Dense(1, activation='sigmoid'))
   
   model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=4, batch_size=512)
   results = model.evaluate(x_test, y_test)
   print(results)
   
   
   # 층 유닛 32개로 증가
   model = models.Sequential()
   model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
   model.add(layers.Dense(32, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
   
   model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=4, batch_size=512)
   results = model.evaluate(x_test, y_test)
   print(results)
   
   model = models.Sequential()
   model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
   model.add(layers.Dense(16, activation='relu'))
   model.add(layers.Dense(1, activation='sigmoid'))
   
   
   #loss 함수 변경
   model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy']) 
   model.fit(x_train, y_train, epochs=4, batch_size=512)
   results = model.evaluate(x_test, y_test)
   print(results)
   
   
   # loss 함수 변경
   model = models.Sequential()
   model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
   model.add(layers.Dense(16, activation='tanh'))
   model.add(layers.Dense(1, activation='sigmoid'))
   
   model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=4, batch_size=512)
   results = model.evaluate(x_test, y_test)
   print(results)
   ```

   



