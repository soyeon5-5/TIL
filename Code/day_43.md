# 22.03.09

## Tensorflow

### Fitervisualization

- VGG16 모델 사용

```python
from tensorflow.keras.applications import VGG16
model = VGG16(weights = 'imagenet')
```

```python
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
```

1. 이미지 전처리

   ```python
   img_path = 'creative_commons_elephant.jpg'
   
   img = image.load_img(img_path, target_size=(224,224))
   
   x = image.img_to_array(img)
   
   x = np.expand_dims(x,axis=0)
   
   x=preprocess_input(x)
   ```

2. 예측 및 결과

   ```python
   preds = model.predict(x)
   print('Predicted:', decode_predictions(preds, top=3)[0])
   print(np.argmax(preds[0]))
   ```

3. Visualizing heatmap of class activation

   - Grad-Cam 알고리즘 설정

     ```python
     import tensorflow as tf
     tf.compat.v1.disable_eager_execution()
     from keras import backend as K
     
     african_elephant_output = model.output[:, 386]
     last_conv_layer = model.get_layer('block5_conv3')
     grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
     pooled_grads = K.mean(grads, axis=(0, 1, 2))
     iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
     pooled_grads_value, conv_layer_output_value = iterate([x])
     for i in range(512):
       conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
     ```

   - Heatmap post-processing

     ```python
     heatmap = np.mean(conv_layer_output_value, axis=-1)
     heatmap = np.maximum(heatmap, 0)
     heatmap /= np.max(heatmap, 0)
     plt.matshow(heatmap) #386위치에서 활성화되는부분 히트맵
     ```

   - 원본 이미지에 heatmap 덧붙이기

     ```python
     import cv2
     
     img = cv2.imread(img_path)
     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
     heatmap = np.uint8(255 * heatmap)
     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
     superimposed_img = heatmap * 0.4 + img
     
     cv2.imwrite('./elephant_cam.jpg', superimposed_img)
     ```