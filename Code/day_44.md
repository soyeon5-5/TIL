# 22.03.10

## Tensorflow

#### RNN

- 단어 임베딩

1. 데이터 준비

   ```python
   # Imdb 이용
   import os
   
   # !unzip 'aclImdb.zip'
   
   imdb_dir = './aclImdb'
   train_dir = os.path.join(imdb_dir, 'train')
   
   labels = []
   texts = []
   
   
   for label_type in  ['neg', 'pos']:
       dir_name = os.path.join(train_dir, label_type).replace("\\","/")
       
       for fname in os.listdir(dir_name):
           if fname[-4:] == '.txt':
               f = open(os.path.join(dir_name, fname), encoding='utf8')
               texts.append(f.read())
               f.close()
               
               if label_type == 'neg':
                   labels.append(0)
               else:
                   labels.append(1)
   ```

   ```python
   # 텍스트를 벡터로
   from keras.preprocessing.text import Tokenizer
   from keras.preprocessing.sequence import pad_sequences
   import numpy as np
   
   maxlen =100
   training_samples =200
   validation_samples=10000
   max_words =10000
   
   tokenizer = Tokenizer(num_words=max_words)
   tokenizer.fit_on_texts(texts)
   sequences = tokenizer.texts_to_sequences(texts)
   
   word_index = tokenizer.word_index
   print('Found %s unique tokens.' % len(word_index))
   
   data = pad_sequences(sequences, maxlen=maxlen)
   labels = np.asarray(labels)
   print('데이터 텐서의 크기:', data.shape)
   print('레이블 텐서의 크기:', labels.shape)
   
   indices = np.arange(data.shape[0])
   np.random.shuffle(indices)
   data= data[indices]
   labels = labels[indices]
   
   x_train = data[:training_samples]
   y_train = labels[:training_samples]
   x_val = data[training_samples: training_samples + validation_samples]
   y_val = labels[training_samples : training_samples+validation_samples]
   ```

   ```python
   # glove.6B.zip 사용
   from keras.datasets import imdb
   from keras import preprocessing
   import os
   import numpy as np
   
   #!unzip 'glove.6B.100d.txt.zip'
   ```

   ```python
   # 압축해제한 txt 파싱하여 단어와 상응하는 벡터 표현 매핑하는 인덱스
   glove_dir = './glove.6B.100d.txt'
   
   
   embeddings_index = {} 
   f = open(os.path.join(glove_dir))
   for line in f: 
     values = line.split() 
     word = values[0] 
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs 
   f.close() 
   
   print('Found %s word vectors.' % len(embeddings_index))
   ```

   ```python
   # 임베딩 행렬 생성
   embedding_dim = 100 
   embedding_matrix = np.zeros((max_words, embedding_dim))
   for word, i in word_index.items(): 
     if i < max_words: 
       embedding_vector = embeddings_index.get(word)
       if embedding_vector is not None: 
         embedding_matrix[i] = embedding_vector
   ```

2. 모델 구성

   ```python
   from keras.models import Sequential
   from keras.layers import Embedding, Flatten, Dense
   
   model = Sequential()
   
   model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
   model.add(Flatten())
   model.add(Dense(32, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   
   model.summary()
   ```

   ```python
   # 임베딩층은 동결
   model.layers[0].set_weights([embedding_matrix])
   model.layers[0].trainable= False
   
   model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
   history=model.fit(x_train,y_train,
                     epochs=10,
                     batch_size=32,
                     validation_data=(x_val,y_val))
   model.save_weights('pre_trained_glove_model.h5')
   ```

3. 그래프 확인

   ```python
   import matplotlib.pyplot as plt
   
   acc = history.history['acc']
   val_acc = history.history['val_acc']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   
   epochs= range(1, len(acc)+1)
   
   plt.plot(epochs,acc, 'bo', label='Training acc')
   plt.plot(epochs, val_acc, 'b', label='Validation acc')
   plt.title('Training and validation accuracy')
   plt.legend()
   
   plt.figure()
   
   plt.plot(epochs, loss, 'bo', label='Training loss')
   plt.plot(epochs, val_loss, 'b', label='Validation loss')
   plt.title('Training and validation loss') 
   plt.legend()
   
   plt.show()
   
   # 과적합이 빠르게 시작 -> 훈련 샘플수가 작아서
   ```


- Withdout pretrained word embeddings

  - 모델 구성

    ```python
    from keras.models import Sequential 
    from keras.layers import Embedding, Flatten, Dense
    
    model = Sequential() 
    
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten()) 
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    ```

    ```python
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    
    history = model.fit(x_train, y_train, 
     epochs=10, 
     batch_size=32, 
     validation_data=(x_val, y_val))
    ```

  - 그래프 확인

    ```python
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs= range(1, len(acc)+1)
    
    plt.plot(epochs,acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss') 
    plt.legend()
    
    plt.show()
    ```

  - test set으로

    ```python
    test_dir = os.path.join(imdb_dir, 'test')
    
    labels=[]
    texts=[]
    
    for label_type in ['neg', 'pos']: 
      dir_name = os.path.join(test_dir, label_type)
      for fname in sorted(os.listdir(dir_name)): 
        if fname[-4:] == '.txt': 
          f = open(os.path.join(dir_name, fname))
          texts.append(f.read()) 
          f.close() 
          if label_type == 'neg': 
            labels.append(0)
          else: 
            labels.append(1)
    
    sequences= tokenizer.texts_to_sequences(texts)
    x_test = pad_sequences(sequences, maxlen = maxlen)
    y_test = np.asarray(labels)
    ```

    ```python
    # 평가
    model.load_weights('pre_trained_glove_model.h5')
    model.evaluate(x_test, y_test)
    ```

- SimpleRNN

  ```python
  # output으로 last timestep만
  from keras.models import Sequential
  from keras.layers import Embedding, SimpleRNN
  model = Sequential()
  model.add(Embedding(10000, 32))
  model.add(SimpleRNN(32))
  model.summary()
  ```
  
  ```python
  # full state sequence
  model = Sequential()
  model.add(Embedding(10000, 32))
  model.add(SimpleRNN(32, return_sequences=True))
  model.summary()
  ```
  
  ```python
  # 네트워크의 표현력 증가를 위해 여러개의 순환층을 차례대로 쌓기 -> 중간층들이 전체 출력 시퀀스를 반환하도록 설정
  
  model = Sequential()
  model.add(Embedding(10000, 32))
  model.add(SimpleRNN(32, return_sequences=True))
  model.add(SimpleRNN(32, return_sequences=True))
  model.add(SimpleRNN(32, return_sequences=True))
  model.add(SimpleRNN(32))
  model.summary()
  ```
  
  - IMDB 영화리뷰 분류에 적용
  
    1. 데이터 준비
  
       ```python
       from keras.datasets import imdb 
       from keras.preprocessing import sequence 
       
       max_features = 10000 
       maxlen = 500 
       batch_size = 32 
       
       print('Loading data...') 
       (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features) 
       print(len(input_train), 'train sequences') 
       print(len(input_test), 'test sequences')
       
       
       print('Pad sequences (samples x time)') 
       input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
       input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
       print('input_train shape:', input_train.shape) 
       print('input_test shape:', input_test.shape)
       ```
  
    2. 모델 구성
  
       ```python
       from keras.layers import Dense 
       
       model = Sequential()
       model.add(Embedding(max_features, 32))
       model.add(SimpleRNN(32))
       model.add(Dense(1, activation='sigmoid')) 
       
       model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
       history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
       ```
  
    3. 그래프 비교
  
       ```python
       import matplotlib.pyplot as plt
       
       acc = history.history['acc'] 
       val_acc = history.history['val_acc']
       loss = history.history['loss']
       val_loss = history.history['val_loss']
       
       epochs = range(1, len(acc) + 1)
       
       plt.plot(epochs, acc, 'bo', label='Training acc')
       plt.plot(epochs, val_acc, 'b', label='Validation acc') 
       plt.title('Training and validation accuracy')
       plt.legend()
       
       plt.figure()
       
       plt.plot(epochs, loss, 'bo', label='Training loss')
       plt.plot(epochs, val_loss, 'b', label='Validation loss')
       plt.title('Training and validation loss') 
       plt.legend() 
       plt.show()
       # SimpleRNN이 텍스트처럼 긴 시퀀스를 사용하는데 적합하지 않음
       ```
  
       



