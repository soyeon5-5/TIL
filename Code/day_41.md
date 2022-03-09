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

   ```python
   train_images, test_images = train_images /255.0, test_images/255.0
   ```
   
   ```python
   class_names = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
   ```
   
   ```python
   # 데이터 확인
   plt.figure(figsize=(10,10))
   for i in range(25):
     plt.subplot(5,5,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(train_images[i], cmap=plt.cm.binary)
     plt.xlabel(class_names[train_labels[i][0]])
   plt.show()
   ```
   
3. 모델 구성

   ```python
   model = models.Sequential()
   model.add(layers.Conv2D(32,(3,3,), activation='relu', input_shape=(32,32,3)))
   model.add(layers.MaxPooling2D(2,2))
   model.add(layers.Conv2D(64,(3,3,), activation='relu'))
   model.add(layers.MaxPooling2D(2,2,))
   model.add(layers.Conv2D(64,(3,3), activation='relu'))
   
   model.summary()
   ```

   ```python
   # dense 추가
   model.add(layers.Flatten())
   model.add(layers.Dense(64, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))
   ```

   ```python
   model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
   
   history=model.fit(train_images, train_labels, epochs=10,
                      validation_data=(test_images, test_labels))
   ```

4. 그래프 비교

   ```python
   plt.plot(history.history['accuracy'], label='accuracy')
   plt.plot(history.history['val_accuracy'], label='val_accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.ylim([0.5 ,1])
   plt.legend(loc='lower right')
   
   test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
   ```



---

#### MNIST 손글씨

1. 데이터 준비

   ```python
   from numpy import mean
   from numpy import std
   from matplotlib import pyplot
   from sklearn.model_selection import KFold
   from keras.datasets import mnist
   from tensorflow.keras.utils import to_categorical
   from keras.models import Sequential
   from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
   from tensorflow.keras.optimizers import SGD
   ```

   ```python
   # 데이터 준비 함수
   def load_dataset():
       (trainX, trainY), (testX,testY) = mnist.load_data()
     trainX = trainX.reshape((trainX.shape[0], 28, 28,1))
     testX = testX.reshape((testX.shape[0],28,28,1))
   
     trainY=to_categorical(trainY)
     testY=to_categorical(testY)
     return trainX, trainY, testX, testY
   ```

2. 데이터 전처리

   ```python
   # 데이터 전처리 함수
   # scale pixels
   def prep_pixels(train, test):
     #convert from integers to floats
     train_norm = train.astype('float32')
     test_norm = test.astype('float32')
     # normalize to range(0-1)
     train_norm= train_norm/255.0
     test_norm = test_norm/255.0
     return train_norm, test_norm
   ```

3. 모델 구성

   ```python
   # 모델 함수
   # define cnn model
   def define_model():
     model = Sequential()
     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
     model.add(MaxPooling2D((2, 2)))
     model.add(Flatten())
     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
     model.add(Dense(10, activation='softmax'))
   # compile model
     opt = SGD(learing_rate=0.01, momentum=0.9)
     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
     return model
   ```

   ```python
   # evaluate, using k-fold cross validation
   def evaluate_model(dataX, dataY, n_folds = 5):
     scores, histories = list(), list()
   
     kfold = KFold(n_folds, shuffle = True, random_state = 1)
   
     # enumerate splits
     for train_ix, test_ix in kfold.split(dataX):
       model = define_model()
       trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
       history = model.fit(trainX, trainY, epochs = 10, batch_size = 32, validation_data = (testX, testY), verbose = 0)
       _, acc = model.evaluate(testX, testY, verbose = 0)
       print('> %.3f' %(acc * 100.0))
       scores.append(acc)
       histories.append(history)
     return scores, histories
   ```

4. 그래프

   ```python
   # 그래프 비교 함수
   def summarize_diagnostics(histories):
     for i in range(len(histories)):
       pyplot.subplot(2, 1, 1)
       pyplot.title('Cross Entropy Loss')
       pyplot.plot(histories[i].history['loss'], color = 'blue', label = 'train')
       pyplot.plot(histories[i].history['val_loss'], color = 'orange', label = 'test')
   
       pyplot.subplot(2, 1, 2)
       pyplot.title('Classification Accuracy')
       pyplot.plot(histories[i].history['accuracy'], color = 'blue', label = 'train')
       pyplot.plot(histories[i].history['val_accuracy'], color = 'orange', label = 'test')
     pyplot.show()
   
   # summarize model performance
   def summarize_performance(scores):
     print('Accuracy : mean = %.3f, std = %.3f, n = %d' %(mean(scores) * 100, std(scores) * 100, len(scores)))
     pyplot.boxplot(scores)
     pyplot.show()
   ```

5. 함수 실행

   ```python
   def run_test_harness():
     trainX, trainY, testX, testY = load_dataset()
     trainX, testX = prep_pixels(trainX, testX)
     scores, histories = evaluate_model(trainX, trainY)
     summarize_diagnostics(histories)
     summarize_performance(scores)
   
   run_test_harness()
   ```

   