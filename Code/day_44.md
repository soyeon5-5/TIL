# 22.03.10

## Tensorflow

#### RNN

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
   ```

   ```python
   data = pad_sequences(sequences, maxlen=maxlen)
   labels = np.asarray(labels)
   print('데이터 텐서의 크기:', data.shape)
   print('레이블 텐서의 크기:', labels.shape)
   ```

   ```python
   indices = np.arange(data.shape[0])
   np.random.shuffle(indices)
   data= data[indices]
   labels = labels[indices]
   
   x_train = data[:training_samples]
   y_train = labels[:training_samples]
   x_val = data[training_samples: training_samples + validation_samples]
   y_val = labels[training_samples : training_samples+validation_samples]
   ```

   