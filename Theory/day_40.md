# 22.03.04

## TensorFlow

```python
pip install tensorflow
```

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

   



