# 22.03.08

## Tensorflow

### Cats and Dog

1. 데이터 준비

   > 이미지 다운받은 후에 # 포함 모두 해야함

   ```python
   import os, shutil
   
   # 경로 설정
   original_db_dir = './train'
   
   base_dir = './cats_and_dogs_small'
   #os.mkdir(base_dir)
   train_dir = os.path.join(base_dir, 'train')
   #os.mkdir(train_dir)
   validation_dir = os.path.join(base_dir, 'validation')
   #os.mkdir(validation_dir)
   test_dir = os.path.join(base_dir, 'test')
   #os.mkdir(test_dir)
   
   train_cats_dir = os.path.join(train_dir, 'cats')
   #os.mkdir(train_cats_dir)
   train_dogs_dir = os.path.join(train_dir, 'dogs')
   #os.mkdir(train_dogs_dir)
   
   validation_cats_dir = os.path.join(validation_dir, 'cats')
   #os.mkdir(validation_cats_dir)
   validation_dogs_dir = os.path.join(validation_dir, 'dogs')
   #os.mkdir(validation_dogs_dir)
   
   test_cats_dir = os.path.join(test_dir, 'cats')
   #os.mkdir(test_cats_dir)
   test_dogs_dir = os.path.join(test_dir, 'dogs')
   #os.mkdir(test_dogs_dir
   ```

   ```python
   #frames = ['cat.{}.jpg'.format(i) for i in range(1000)]
   #for fname in frames:
   # src = os.path.join(original_db_dir, fname)
   # dst = os.path.join(train_cats_dir, fname)
   # shutil.copyfile(src,dst)
   
   #frames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
   #for fname in frames:
   # src = os.path.join(original_db_dir, fname)
   # dst = os.path.join(validation_cats_dir, fname)
   # shutil.copyfile(src,dst)
   
   #frames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
   #for fname in frames:
   # src = os.path.join(original_db_dir, fname)
   # dst = os.path.join(test_cats_dir, fname)
   # shutil.copyfile(src,dst)
   
   
   #frames = ['dog.{}.jpg'.format(i) for i in range(1000)]
   #for fname in frames:
   #  src = os.path.join(original_db_dir, fname)
   #  dst = os.path.join(train_dogs_dir, fname)
   #  shutil.copyfile(src, dst)
   
   #frames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
   #for fname in frames:
   #  src = os.path.join(original_db_dir, fname)
   #  dst = os.path.join(train_dogs_dir, fname)
   #  shutil.copyfile(src, dst)
   
   #frames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
   #for fname in frames:
   #  src = os.path.join(original_db_dir, fname)
   #  dst = os.path.join(train_dogs_dir, fname)
   #  shutil.copyfile(src, dst)
   ```

#### CNN

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import models
```

1. 모델 구성

   ```python
   model = models.Sequential()
   model.add(Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
   model.add(MaxPooling2D(2,2))
   model.add(Conv2D(64,(3,3), activation='relu'))
   model.add(MaxPooling2D(2,2))
   model.add(Conv2D(128,(3,3), activation='relu'))
   model.add(MaxPooling2D(2,2))
   model.add(Conv2D(128,(3,3), activation='relu'))
   model.add(MaxPooling2D(2,2))
   model.add(Flatten())
   model.add(Dense(512, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   
   model.summary()
   
   from tensorflow.keras import optimizers
   
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])
   ```

2. ImageDataGenerator

   ```python
   from keras.preprocessing.image import ImageDataGenerator
   ```

   ```python
   train_datagen = ImageDataGenerator(rescale=1./255)
   test_datagen = ImageDataGenerator(rescale=1./255)
   
   train_generator  = train_datagen.flow_from_directory(
       train_dir,
       target_size=(150,150),
       batch_size=20,
       class_mode='binary'
   )
   
   validation_generator  = test_datagen.flow_from_directory(
       validation_dir,
       target_size=(150,150),
       batch_size=20,
       class_mode='binary'
   )
   ```

   ```python
   for data_batch, labels_batch in train_generator:
     print('배치 데이터 크기:', data_batch.shape)
     print('배치 레이블 크기:', labels_batch.shape)
     break
   ```

3. 모델 훈련

   ```python
   history = model.fit_generator(
       train_generator,
       steps_per_epoch=100,
       epochs=30,
       validation_data=validation_generator,
       validation_steps=50
   )
   
   model.save('./drive/MyDrive/cats_and_dogs_small_1.h5')
   ```

4. 그래프 비교

   ```
   import matplotlib.pyplot as plt
   
   acc=history.history['acc']
   val_acc = history.history['val_acc']
   loss=history.history['loss']
   val_loss = history.history['val_loss']
   
   epochs=range(1, len(acc)+1)
   
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
   ```

   

#### Data Augmentation

```python
### 연습 예시
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

from keras.preprocessing import image

fnames=sorted([os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)])
img_path = fnames[3]
img=image.load_img(img_path, target_size=(150,150))
x=image.img_to_array(img)
x=x.reshape((1,)+x.shape)
i=0
for batch in datagen.flow(x, batch_size=1):
  plt.figure(i)
  imgplot = plt.imshow(image.array_to_img(batch[0]))
  i += 1
  if i%4 == 0:
    break
plt.show()
```

1. 모델 구성

   ```python
   model = models.Sequential()
   model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
   model.add(MaxPooling2D((2, 2)))
   model.add(Conv2D(64, (3, 3), activation = 'relu'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Conv2D(128, (3, 3), activation = 'relu'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Conv2D(128, (3, 3), activation = 'relu'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Flatten())
   model.add(Dropout(0.5))
   model.add(Dense(512, activation = 'relu'))
   model.add(Dense(1, activation = 'sigmoid'))
   
   model.summary()
   ```

   ```python
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])
   ```

2. ImageDataGenerator

   ```python
   train_datagen = ImageDataGenerator(
       rescale = 1./255,
       rotation_range = 40,
       width_shift_range = 0.2,
       height_shift_range = 0.2,
       shear_range = 0.2,
       zoom_range = 0.2,
       horizontal_flip = True,
   )
   test_datagen = ImageDataGenerator(rescale = 1./255)
   
   train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size = (150, 150),
       batch_size = 32,
       class_mode = 'binary'
   )
   
   validation_generator = test_datagen.flow_from_directory(
       validation_dir,
       target_size = (150, 150),
       batch_size = 32,
       class_mode = 'binary'
   )
   ```

3. 모델 훈련

   ```python
   history = model.fit_generator(
       train_generator,
       #steps_per_epoch=100, # 에러시 줄이거나 생략
       epochs=300,
       validation_data=validation_generator,
       #validation_steps=50  # 에러시 생략
   )
   
   
   model.save('./drive/MyDrive/cats_and_dogs_small_2.h5')
   ```

4. 그래프 비교

   ```python
   acc = history.history['acc']
   val_acc = history.history['val_acc']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   epochs = range(1, len(acc)+1)
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
   ```



#### Transfer Learning

> 특성 추출,  미세 조정

1. 특성 추출(Feature Extraction)

   ```python
   from tensorflow.keras.applications.vgg16 import VGG16
   
   conv_base = VGG16(weights = 'imagenet',
                    include_top = False,
                    input_shape=(150,150,3))
   
   conv_base.summary
   ```

   1. 데이터 준비

      ```python
      import os
      import numpy as np
      from kears.preprocessing.image import ImageDataGenerator
      ```

      ```python
      !unzip 'cats_and_dogs_small.zip'
      ```

      ```python
      base_dir = './cats_and_dogs_small'
      validation_dir = os.path.join(base_dir, 'validation')
      test_dir = os.path.join(base_dir, 'test')
      
      datagen = ImageDataGenerator(rescale=1./255)
      batch_size =20
      ```

      
