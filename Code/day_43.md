# 22.03.09

## Tensorflow

### Fitervisualization

- VGG16 모델 사용

- 필터 활성화 부분 시각화

  1. 준비

     ```python
     import numpy as np
     import tensorflow as tf
     from tensorflow import keras
     
     # The dimensions of our input image
     img_width = 180
     img_height = 180
     # Our target layer: we will visualize the filters from this layer.
     # See `model.summary()` for list of layer names, if you want to change this.
     layer_name = "conv3_block4_out"
     ```

  2. 모델 구성

     ```python
     # Build a ResNet50V2 model loaded with pre-trained ImageNet weights
     model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
     
     # Set up a model that returns the activation values for our target layer
     layer = model.get_layer(name=layer_name)
     feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
     ```

     ```python
     def compute_loss(input_image, filter_index):
         activation = feature_extractor(input_image)
      	# We avoid border artifacts by only involving non-border pixels in the loss.
      	filter_activation = activation[:, 2:-2, 2:-2, filter_index]
      	return tf.reduce_mean(filter_activation)
      
     @tf.function
     def gradient_ascent_step(img, filter_index, learning_rate):
         with tf.GradientTape() as tape:
      	tape.watch(img)
         loss = compute_loss(img, filter_index)
         # Compute gradients.
         grads = tape.gradient(loss, img)
         # Normalize gradients.
         grads = tf.math.l2_normalize(grads)
         img += learning_rate * grads
         return loss, img
     
     def initialize_image():
         # We start from a gray image with some random noise
         img = tf.random.uniform((1, img_width, img_height, 3))
         # ResNet50V2 expects inputs in the range [-1, +1].
         # Here we scale our random inputs to [-0.125, +0.125]
         return (img - 0.5) * 0.25
     
     def visualize_filter(filter_index):
         # We run gradient ascent for 20 steps
         iterations = 30
         learning_rate = 10.0
         img = initialize_image()
         for iteration in range(iterations):
             loss, img = gradient_ascent_step(img, filter_index, learning_rate)
             # Decode the resulting input image
             img = deprocess_image(img[0].numpy())
             return loss, img
     
     def deprocess_image(img):
         # Normalize array: center on 0., ensure variance is 0.15
         img -= img.mean()
         img /= img.std() + 1e-5
         img *= 0.15
         # Center crop
         img = img[25:-25, 25:-25, :]
         # Clip to [0, 1]
         img += 0.5
         img = np.clip(img, 0, 1)
         # Convert to RGB array
         img *= 255
         img = np.clip(img, 0, 255).astype("uint8")
         return img
     ```

  3. 활성화 이미지 시각화

     ```python
     # 필터 0위치
     from IPython.display import Image, display
     
     loss, img = visualize_filter(0)
     keras.preprocessing.image.save_img("0.png", img)
     
     display(Image("0.png"))
     ```

     ```python
     # 필터 전체 위치 시각화
     # Compute image inputs that maximize per-filter activations
     # for the first 64 filters of our target layer
     all_imgs = []
     for filter_index in range(64):
         print("Processing filter %d" % (filter_index,))
         loss, img = visualize_filter(filter_index)
         all_imgs.append(img)
     # Build a black picture with enough space for
     # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
     margin = 5
     n = 8
     cropped_width = img_width - 25 * 2
     cropped_height = img_height - 25 * 2
     width = n * cropped_width + (n - 1) * margin
     height = n * cropped_height + (n - 1) * margin
     stitched_filters = np.zeros((width, height, 3))
     
     # Fill the picture with our saved filters
     for i in range(n):
         for j in range(n):
             img = all_imgs[i * n + j]
             stitched_filters[
                 (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                 (cropped_height + margin) * j : (cropped_height + margin) * j
                 + cropped_height,
                 :,
             ] = img
     keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)
     
     from IPython.display import Image, display
     
     display(Image("stiched_filters.png"))
     ```

- 실제 사진과 활성화 부분 시각화

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