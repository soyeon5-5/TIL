# 22.01.24

## Open CV

```python
# ! pip install opencv-python
import cv2
import numpy as np
import sys
```

#### 1. 이미지 열기

```python
img = cv2.imread('./fig/puppy.bmp', cv2.IMRAD_COLOR)
# print(type(img)) = numpy.array임

if img is None :
    print('image read failed')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)

# 이 코드가 없으면 아무키 눌러 종료
# 반드시 키값으로 꺼야함, 안그러면 계속 런
while True :
    key = cv2.waitKey()
    # 27은 ascii 코드로 esc 의미 / ord()는 ascii로 변환
    if key == 27 or ord('a')
    	break

cv2.destroyAllWindows()
```

- 사이즈 조절

  ```python
  # 1. 마우스로 이미지 창 사이즈 조절
  img = cv2.imread('./fig/puppy_1280_853.jpg', cv2.IMREAD_COLOR)
  
  if img is None :
      print('image read failed')
      sys.exit()
  
  # 마우스로 크기조절가능해짐
  cv2.namedWindow('image', cv2.WINDOW_NORMAL) 
  
  #뜨는 위치 조절
  cv2.moveWindow('image', 0, 200)
  
  cv2.imshow('image', img)
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  
  
  # 2. resize 해주기
  # image dimension = (480, 640) 이지만
  # numpy 형식으로 변환되므로 (320, 240)으로 해줘야함
  img = cv2.imread('./fig/puppy_1280_853.jpg', cv2.IMREAD_COLOR)
  img_re = cv2.resize(img, (320, 240), cv2.INTER_AREA)
  
  if img is None :
      print('image read failed')
      sys.exit()
      
  cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('image_re', img_re)
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  
  
  #3. imread에서 reduce
  img = cv2.imread('./fig/puppy_1280_853.jpg',
                  flags = cv2.IMREAD_REDUCED_COLOR_4)
  if img is None :
      print('image read failed')
      sys.exit()
      
  cv2.namedWindow('image')
  cv2.imshow('image', img)
  cv2.waitKey()
  
  cv2.destroyAllWindows()
  ```

- Matplotlib 으로 보기

  ```python
  import matplotlib.pyplot as plt
  ```

  ```python
  img  = cv2.imread('./fig/puppy.bmp', cv2.IMREAD_COLOR)
  # BGR 형태
  
  if img is None :
      print('image read failed')
      sys.exit()
  
  #BGR -> RGB
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  #BGR -> Gray
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  #plt로 동시
  plt.figure(figsize = (12, 6))
  plt.subplot(131), plt.imshow(img), plt.axis('off')
  plt.subplot(132), plt.imshow(imgRGB), plt.axis('off')
  plt.subplot(133), plt.imshow(imgGray, cmap = 'gray'), plt.axis('off')
  plt.show()
  ```

- 이미지 슬라이드

  ```python
  img_path =[]
  for i in img_list :
      img_path_all = './fig/images/' + i
      img_path.append(img_path_all)
  
  
  cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)
  cnt = len(img_path)
  idx = 0
  
  while True:
      img_name = img_path[idx]
      img = cv2.imread(img_name, cv2.IMREAD_COLOR)
      
      cv2.imshow('image', img)
      
      if cv2.waitKey(1000) == 27:
          break
      
      idx += 1
      if idx >= cnt:
          idx = 0
          
  cv2.destroyAllWindows()
  ```

  
