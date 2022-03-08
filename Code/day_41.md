# 22.03.07

## Neural Network

#### 뉴스 기사 분류 - 다중 분류

1. 데이터 준비

   ```python
   from keras.datasets import reuters
   
   (train_data, train_labels), (test_data, test_labels)= reuters.load_data(num_words=10000)
   ```

2. 데이터 전처리

   ```python
   # 데이터를 벡터로 변환
   import numpy as np
   
   def vectorize_sequences(seqs, dim=10000):
       results = np.zeros((len(seqs), dim))
       for i, seq in enumerate(seqs):
           results[i, seq]= 1.
       return results
   
   x_train = vectorize_sequences(train_data)
   x_test = vectorize_sequences(test_data)
   ```

   ```python
   # 레이블을 벡터로 변환
   # one-hot encoding이 범주형 데이터에 널리 사용
   
   def to_one_hot(labels, dim=46):
       results = np.zeros((len(labels), dim))
       for i, l in enumerate(labels):
           results[i, l] =1.
       return results
   
   one_hot_train_labels = to_one_hot(train_labels)
   one_hot_test_labels = to_one_hot(test_labels)
   
   # 다른 방법
   # from keras.utils.np_utils import to_categorical
   # one_hot_train_labels = to_categorical(train_labels)
   # one_hot_test_lables = to_categorical(test_labels)
   ```

3. 모델 구성

   ```python
   from keras import models
   from keras import layers
   
   model = models.Sequentail()
   model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(46, activation='softmax')) # 각 클래스에 대한 확률분포 출력 46개의 값 모두 더하면 1
   ```

   ```python
   # 이때 최선의 손실함수 categorical_crossentropy
   model.compile(optimizer='rmrprop', loss='categorical_crossentropy', metrics=['accuracy'])
   ```

   ```python
   # 훈련
   # 훈련데이터 중 일부 검증세트로 사용
   x_val = x_train[:1000]
   partial_x_train = x_train[1000:]
   y_val = one_hot_train_labels[:1000]
   partial_y_train = one_hot_train_labels[1000:]
   ```

   ```python
   history = model.fit(partial_x_train,partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
   ```

4. 그래프 비교

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
   acc = history_dict['accuracy']
   val_acc = history_dict['val_accuracy']
   
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
   model = models.Sequentail()
   model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(46, activation='softmax'))
   
   model.compile(optimizer='rmrprop', loss='categorical_crossentropy', metrics=['accuracy'])
   model.fit(partial_x_train,partial_y_train, epochs=9, batch_size=512)
   results = model.evaluate(x_test, one_hot_test_labels)
   print(results)
   ```

6. 예측

   ```python
   predictions = model.predict(x_test)
   print(predictions[0].shape)
   print(np.sum(predictions[0]))
   print(np.argmax(predictions[0]))
   ```



---

## CNN

Convolution Neural Network

#### 이미지 분류

1. 데이터 준비

   ```
   import tensorflow as tf
   
   from tensorflow.keras import datasets, layers, models
   import matplotlib.pyplot as plt
   
   (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
   ```

2. 데이터 전처리

   ```
   ```

   