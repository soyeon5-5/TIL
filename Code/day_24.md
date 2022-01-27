# 22.01.25

## Open CV basic

```python
import numpy as np
import cv2
import sys
```

###  합성

- 영상 크기 참조

  ```python
  img = cv2.imread('./fig/puppy.bmp', cv2.IMREAD_GRAYSCALE)
  
  if img is None :
      print('image read failed')
      sys.exit
      
  print('img type = ', type(img))
  
  print('img dimension = ', img.shape)
  
  h, w =img.shape[:2] # 컬러의 경우 dimesino값이 따라올수있으므로
  print('img size = {} x {}'.format(w, h))
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  ```

- 영상 픽셀값 참조

  ```python
  img1 = cv2.imread('./fig/puppy.bmp', 1)
  
  if img1 is None or img2 is None:
      print('image read failed')
      sys.exit
      
  # 영상의 센터 찾기
  h, w = img1.shape[:2]
  img1_center = img1[h//2, w//2]
  
  # 부분 값 변경
  img1[10:110, 100:200] = [0, 255, 0] # bgr - green만 , list형태나, tuple 형태나 상관없음
  
  cv2.imshow('image1', img1)
  
  cv2.waitKey()
  
  cv2.destroyAllWindows()
  ```

- 영상 생성

  ```python
  img1 = np.zeros((240, 320, 3), dtype = np.uint8) # ones : 전체 0
  img2 = np.ones((240, 320, 3), dtype = np.uint8) * 255  # ones : 전체 1
  img3 = np.full((240, 320, 3), 255, dtype = np.uint8) # 원하는색으로 full
  img4 = np.random.randint(0, 255, size = (240, 320, 3), dtype = np.uint8) # 랜덤색상
  
  img1[10:60, 10:60] = (0, 0, 255)
  
  cv2.imshow('img1', img1)
  cv2.imshow('img2', img2)
  cv2.imshow('img3', img3)
  cv2.imshow('img4', img4)
  
  cv2.waitKey()
  
  cv2.destroyAllWindows()
  ```
  
- 영상 복사

  ```python
  img = cv2.imread('./fig/cat.bmp')
  
  if img is None :
      print('image read failed')
      sys.exit
  
  img1 = img  # 주소가 같음, 복사x
  img2 = img.copy()  # 주소가 변경됨, 복사, 백업 하기
  
  
  img1[100:200, 200:300] = (0, 255, 255)
  
  cv2.imshow('image', img)
  cv2.imshow('image1', img1)
  cv2.imshow('image2', img2)
  
  while True :
      key = cv2.waitKey()
      if key == 27 :
          break
  
  cv2.destroyAllWindows()
  # img, img1은 같은 영상, img2는 원본상태 그대로
  ```

  - 원 그려넣기

    ```python
    # LINE_AA :라인을 smooth 하게 해줌
    # thickness 부분에 -1 넣으면 원 다 채워짐
    cv2.circle(image, (100, 200), 100, (0, 0, 255), 3, cv2.LINE_AA)
    
    cv2.imshow('image', image)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    ```

- **copyTo**

  ```python
  src = cv2.imread('./fig/airplane.bmp', cv2.IMREAD_COLOR)
  mask = cv2.imread('./fig/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
  dst = cv2.imread('./fig/field.bmp', cv2.IMREAD_COLOR)
  
  if src is None or mask is None or dst is None :
      print('image read failed')
      sys.exit()
  
  # mask에서 0이 아닌부분만 dst에 넣음, dimension도 같아야함
  cv2.copyTo(src, mask, dst)
  # dst = cv2.copyTo(src, mask) - 마스크 값에 해당하는 부분이 dst에 대입
  
  
  ## 인덱스 이용해서도 copyTo 와 같이 가능
  ## dst[mask > 0] = src[mask > 0]
  ## 색상 변경도 가능
  ## dst[mask > 0] = (0, 0, 255)
  
  cv2.imshow('src', src)
  # cv2.imshow('mask', mask)
  cv2.imshow('dst', dst)
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  ```



#### 영상 두개 합성

```python
# 1. scr, dst로 쓸 영상 불러오기
img1 = cv2.imread('./fig/cow.png')
img2 = cv2.imread('./fig/green.png')

# 2. 영상간 크기 맞추기
h, w = img1.shape[:2]
img2_seg = img2[350:350+h, 200:200+w]

# 3. scr 영상 thresholding
# grayscale로 변경
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img1_gray, 240, 255, cv2.THRESH_BINARY_INV)

# 4. copyTo
cv2.copyTo(img1, mask, img2_seg)

# 5. img2의 사이즈 조절(선택)
img2_re = cv2.resize(img2, (1200, 700), cv2.INTER_AREA)

# 6. 확인
cv2.imshow('img1', img1)
cv2.imshow('img2_re', img2_re)
cv2.imshow('mask', mask)
cv2.imshow('img2_seg', img2_seg)

# 7. 저장
cv2.imwrite('./fig/cowingreen.png', img2_re)

cv2.waitKey()
cv2.destroyAllWindows()
```



