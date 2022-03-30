# 22.03.10

## Tensorflow

#### Advanced use of RNN

- 날짜 시계열 데이터로 24시간 후 온도 예측

  1. 데이터 준비

     ```python
     # jena_climate_2009_2016.csv.zip 다운받아오기
     !unzip 'jena_climate_2009_2016.csv.zip'
     ```

     ```python
     import os
     
     data_dir = './jena_climate'
     fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
     
     f = open(fname)
     data = f.read()
     f.close()
     
     lines = data.split('\n')
     header = lines[0].split(',')
     lines = lines[1:]
     
     print(header) 
     print(len(lines))
     ```

     ```python
     #parsing
     import numpy as np
     
     float_data = np.zeros((len(lines), len(header) - 1))
     
     for i, line in enumerate(lines): 
         values = [float(x) for x in line.split(',')[1:]]
         float_data[i, :] = values
         
     # 시계열 데이터 확인
     from matplotlib import pyplot as plt
     
     temp = float_data[:, 1] 
     plt.plot(range(len(temp)), temp)
     
     # 10일간 데이터
     plt.plot(range(1440), temp[:1440])
     ```

  2. 데이터 전처리

     ```python
     mean = float_data[:200000].mean(axis=0)
     float_data -= mean 
     std = float_data[:200000].std(axis=0)
     float_data /= std
     ```

     ```python
     def generator(data, lookback, delay, min_index, max_index,
                   shuffle=False, batch_size=128, step=6):
       if max_index is None:
         max_index = len(data) - delay -1
       i = min_index +lookback
       while 1:
         if shuffle:
           rows = np.random.randint(min_index +lookback, max_index, size = batch_size)
         else:
           if  i +batch_size >= max_index:
             i = min_index+lookback
           rows = np.arange(i, min(i+batch_size, max_index))
           i += len(rows)
         samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
         targets = np.zeros((len(rows),))
         for j, row in enumerate(rows):
           indices = range(rows[j] - lookback, rows[j], step)
           samples[j] = data[indices]
           targets[j] = data[rows[j]+ delay][1]
         yield samples, targets
     ```

     ```python
     from locale import delocalize
     lookback =1440
     step=6
     delay = 144
     batch_size = 128
     
     train_gen = generator(float_data,
                           lookback = lookback,
                           delay=delay,
                           min_index=0,
                           max_index=200000,
                           shuffle=True,
                           step=step,
                           batch_size=batch_size)
     
     val_gen = generator(float_data,
                           lookback = lookback,
                           delay=delay,
                           min_index=200001,
                           max_index=300000,
                           step=step,
                           batch_size=batch_size)
     
     test_gen = generator(float_data,
                           lookback = lookback,
                           delay=delay,
                           min_index=300001,
                           max_index=None,
                           step=step,
                           batch_size=batch_size)
     
     val_steps = (300000-200001 - lookback) //batch_size
     test_steps = (len(float_data)-300001-lookback) //batch_size
     ```

     ```python
     def evaluate_naive_method():
       batch_maes=[]
       for step in range(val_steps):
         samples, targets = next(val_gen)
         preds = samples[:, -1, 1]
         mae = np.mean(np.abs(preds-targets))
         batch_maes.append(mae)
       print(np.mean(batch_maes))
     ```

     ```python
     #계산된 MAE 확인
     print(evaluate_naive_method())
     # 평균 오차
     celsius_mae = 0.29 * std[1]
     ```

  3. 모델 구성

     ```python
     from keras.models import Sequential
     from keras import layers 
     from tensorflow.keras.optimizers import RMSprop 
     
     model = Sequential() 
     model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
     model.add(layers.Dense(32, activation='relu')) 
     model.add(layers.Dense(1))
     
     model.compile(optimizer=RMSprop(), loss='mae')
     
     history = model.fit_generator( train_gen, 
                                   steps_per_epoch=500,
                                   epochs=20,
                                   validation_data=val_gen,
                                   validation_steps=val_steps )
     ```

  4. 그래프 확인

     ```python
     import matplotlib.pyplot as plt 
     
     loss = history.history['loss']
     val_loss = history.history['val_loss'] 
     
     epochs = range(1, len(loss) + 1)
     
     plt.figure() 
     
     plt.plot(epochs, loss, 'bo', label='Training loss')
     plt.plot(epochs, val_loss, 'b', label='Validation loss')
     plt.title('Training and validation loss') 
     plt.legend() 
     plt.show()
     
     # 이 경우는 시계열 데이터를 펼쳐 입력데이터가 시간개념을 잃어버림 -> 순서가 의미있는 시퀀스 데이터 그대로 사용
     # LSTM -> GRU
     ```

  5. 시퀀스 데이터 그대로 다시 모델 구성

     ```python
     from keras.models import Sequential
     from keras import layers 
     from tensorflow.keras.optimizers import RMSprop 
     
     model = Sequential() 
     model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
     model.add(layers.Dense(1)) 
     
     model.compile(optimizer=RMSprop(), loss='mae')
     
     history = model.fit_generator(train_gen, 
                                   steps_per_epoch=500, 
                                   epochs=20, 
                                   validation_data=val_gen, 
                                   validation_steps=val_steps)
     ```

  6. 그래프 확인

     ```python
     import matplotlib.pyplot as plt 
     
     loss = history.history['loss']
     val_loss = history.history['val_loss'] 
     
     epochs = range(1, len(loss) + 1)
     
     plt.figure() 
     
     plt.plot(epochs, loss, 'bo', label='Training loss')
     plt.plot(epochs, val_loss, 'b', label='Validation loss')
     plt.title('Training and validation loss') 
     plt.legend() 
     plt.show()
     ```

  7. Dropout mask 적용 모델 구성

     ```python
     # 모든 타임스텝에 동일한 드롭아웃 마스크 적용하여 학습오차 적절하게 전파
     #overfitting 문제 해결을 위해 매번 드롭아웃을 일정하게 줌
     
     model = Sequential()
     model.add(layers.GRU(32,
                          dropout=0.2, 
                          recurrent_dropout=0.2, 
                          input_shape=(None, float_data.shape[-1])))
     model.add(layers.Dense(1))
     
     model.compile(optimizer=RMSprop(), loss='mae')
     
     history = model.fit_generator(train_gen, 
                                   steps_per_epoch=500, 
                                   epochs=40, 
                                   validation_data=val_gen, 
                                   validation_steps=val_steps)
     ```

  8. 그래프 확인

     ```python
     loss = history.history['loss']
     val_loss = history.history['val_loss'] 
     
     epochs = range(1, len(loss) + 1)
     
     plt.figure() 
     
     plt.plot(epochs, loss, 'bo', label='Training loss')
     plt.plot(epochs, val_loss, 'b', label='Validation loss')
     plt.title('Training and validation loss') 
     plt.legend() 
     plt.show()
     ```

  9. 과대적합이 일어나게 네트워크 용량 늘리기
  
     ```python
     # 층과 유닛수 추가
     from keras.models import Sequential
     from keras import layers 
     from tensorflow.keras.optimizers import RMSprop 
     
     model = Sequential()
     model.add(layers.GRU(32,
                          dropout=0.1, 
                          recurrent_dropout=0.5,
                          return_sequences=True,
                          input_shape=(None, float_data.shape[-1])))
     model.add(layers.GRU(64,
                          activation='relu',
                          dropout=0.1,
                          recurrent_dropout=0.5))
     model.add(layers.Dense(1))
     
     model.compile(optimizer=RMSprop(), loss='mae')
     
     history = model.fit_generator(train_gen, 
                                   steps_per_epoch=500, 
                                   epochs=40, 
                                   validation_data=val_gen, 
                                   validation_steps=val_steps)
     ```
  
  10. 그래프 확인
  
     ```python
     loss = history.history['loss']
     val_loss = history.history['val_loss'] 
     
     epochs = range(1, len(loss) + 1)
     
     plt.figure() 
     
     plt.plot(epochs, loss, 'bo', label='Training loss')
     plt.plot(epochs, val_loss, 'b', label='Validation loss')
     plt.title('Training and validation loss') 
     plt.legend() 
     plt.show()
     # 층 추가에도 성능이 약간 향상될 뿐
     ```

#### Bidirectional RNN

1. 데이터 준비

    ```python
    from keras.datasets import imdb 
    from keras.preprocessing import sequence 
    from keras import layers 
    from keras.models import Sequential

    max_features = 10000 
    maxlen = 500

    (x_train, y_train), (x_test, y_test) = imdb.load_data( num_words=max_features)

    x_train = [x[::-1] for x in x_train] 
    x_test = [x[::-1] for x in x_test]

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    ```

2. 모델 구성

    ```python
    model = Sequential()
    
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
    
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    ```

3. 양방향 모델 구성

   ```python
    model = Sequential()
    model.add(layers.Embedding(max_features, 32))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # LSTM 보다 약간 향상된 것을 확인
   ```

4. 약간의 규제 추가 모델 구성

   ```python
   from keras.models import Sequential
   from keras import layers 
   from keras.optimizers import RMSprop
   
   model = Sequential()
   model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
   model.add(layers.Dense(1))
   
   model.compile(optimizer=RMSprop(), loss='mae')
   history = model.fit_generator(train_gen, 
                                 steps_per_epoch=500, 
                                 epochs=40, 
                                 validation_data=val_gen, 
                                 validation_steps=val_steps)
   ```



