# 22.01.27

## OpenCV

```python
import numpy as np
import cv2
import sys
```

### 1.EVENT

#### 1. 키보드 이벤트

```
img = cv2.imread('./fig/cat.bmp', 0)

if img is None :
    print('image read failed')
    sys.exit()
    
cv2.imshow('image', img)

img1 = img.copy()

while True:
    key = cv2.waitKey()
    if key == 27:
        break
    
    # edge 영상으로 변환
    elif key == ord('e'):
        # 같은 이름으로하면 img가 계속 edge됨
        img = cv2.Canny(img, 50, 150)
        cv2.imshow('image', img)
    
    # inverse 영상으로 변환
    elif key == ord('i'):
        img = 255 - img
        cv2.imshow('image', img)
    
    # original 영상으로 변환
    elif key == ord('r'):
        img = img1.copy()
        cv2.imshow('image', img)
    
        
cv2.destroyAllWindows()
```

#### 2. 마우스 이벤트

```python
#시작포인트
oldx = oldy = 0

def call_mouse(event, x, y, flags, param) :
    global oldx, oldy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 시작점이 누른부분으로
        oldx, oldy = x, y
        print('left btn down = ', x, y)
    
#     elif event == cv2.EVENT_LBUTTONUP :
#         print('left btn up = ', x, y)
        
    elif event == cv2.EVENT_MOUSEMOVE :
        if flags == cv2.EVENT_FLAG_LBUTTON :
            cv2.line(img, (oldx, oldy), (x, y),
                    (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y
    

img = np.ones((480, 640, 3), np.uint8)*255


cv2.namedWindow('image')
cv2.setMouseCallback('image', call_mouse, img)

cv2.imshow('image', img)


while True:
    key = cv2.waitKey()
    if key == 27:
        break
    
    elif key == ord('s'):
        cv2.imwrite('./fig/mysign.png', img)
        
cv2.destroyAllWindows()
```

#### 3. 트랙바

```python
def call_trackbar(pos) :
    global img
    
    img_glass = img *pos
    cv2.imshow('image', img_glass)

    
img_alpha = cv2.imread('./fig/imgbin_sunglasses_1.png', cv2.IMREAD_UNCHANGED)

img = img_alpha[:, :, -1]

img[img > 0] = 1

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.createTrackbar('level', 'image', 0, 255, call_trackbar)

cv2.imshow('image', img)

cv2.waitKey()

cv2.destroyAllWindows()
```



---

### 2. Point processing(영상의 화소처리)

- Sliding

  ```python
  src = cv2.imread('./fig/lenna.bmp', cv2.IMREAD_GRAYSCALE)
  
  if src is None:
      print('image read failed')
      sys.exit()
  
  # 오버 플로우 막기
  #float 연산 후 uint8로 바꿔주기
  # dst = np.clip(src + 100., 0, 255).astype(np.uint8)    
  # -> 한방에 해결
  dst = cv2.add(src, 100)
  
  cv2.imshow('src', src)
  cv2.imshow('dst', dst)
  
  
  cv2.waitKey()
  
  cv2.destroyAllWindows()
  
  
  ## 컬러일때
  src = cv2.imread('./fig/lenna.bmp', cv2.IMREAD_COLOR)
  
  if src is None:
      print('image read failed')
      sys.exit()
  
  # 컬라일때는 3개 다 해주고, 4번째는 알파채널, 
  # 색상부분에 이미지 넣으면 이미지 믹싱
  dst = cv2.add(src, (100, 100, 100, 0)) 
  
  #이건 그냥 다 더해줌
  # dst = np.clip(src + 100, 0, 255).astype(np.uint8) 
  
  cv2.imshow('src', src)
  cv2.imshow('dst', dst)
  
  
  cv2.waitKey()
  
  cv2.destroyAllWindows()
  ```

- 산술연산1

  ```python
  import matplotlib.pyplot as plt
  
  src1 = cv2.imread('./fig/lenna256.bmp', cv2.IMREAD_GRAYSCALE)
  # h, w = src1.shape[:2]
  src2 = np.zeros_like(src1, dtype = np.uint8)
  
  cv2.circle(src2, (128, 128), 100, 200, -1)
  cv2.circle(src2, (128, 128), 50, 50, -1)
  
  dst1 = cv2.add(src1, src2)
  
  # src에 0.5 곱, src2에 0.5 곱, 플러스값, float연산으로!
  dst2 = cv2.addWeighted(src1, 0.7, src2, 0.3, 0.0)
  # 앞-뒤
  dst3 = cv2.subtract(src1, src2)
  # 앞-뒤 절댓값
  dst4 = cv2.absdiff(src1, src2)
  
  plt.figure(figsize = (12, 6))
  plt.subplot(231), plt.imshow(src1, cmap = 'gray'), plt.title('src1'), plt.axis('off')
  plt.subplot(232), plt.imshow(src2, cmap = 'gray'), plt.title('src2'), plt.axis('off')
  plt.subplot(233), plt.imshow(dst1, cmap = 'gray'), plt.title('add'), plt.axis('off')
  plt.subplot(234), plt.imshow(dst2, cmap = 'gray'), plt.title('addweigted'), plt.axis('off')
  plt.subplot(235), plt.imshow(dst3, cmap = 'gray'), plt.title('subtract'), plt.axis('off')
  plt.subplot(236), plt.imshow(dst4, cmap = 'gray'), plt.title('absdiff'), plt.axis('off')
  ```

- 산술연산 2

  ```python
  src1 = np.zeros((256, 256), np.uint8)
  cv2.rectangle(src1, (10, 10), (127, 248), 255, -1)
  
  src2 = np.zeros((256, 256), np.uint8)
  cv2.circle(src2, (128, 128),100, 255, -1)
  
  #픽셀끼리 연산
  dst_bit_and = cv2.bitwise_and(src1, src2)
  dst_bit_or = cv2.bitwise_or(src1, src2)
  
  # 전체-교집합
  dst_bit_xor = cv2.bitwise_xor(src1, src2)
  
  dst_bit_not = cv2.bitwise_not(src1)
  
  
  
  
  plt.figure(figsize = (12, 6))
  plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title('src1')
  plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title('src2')
  plt.subplot(233), plt.axis('off'), plt.imshow(dst_bit_and, 'gray'), plt.title('dst_bit_and')
  plt.subplot(234), plt.axis('off'), plt.imshow(dst_bit_or, 'gray'), plt.title('dst_bit_or')
  plt.subplot(235), plt.axis('off'), plt.imshow(dst_bit_xor, 'gray'), plt.title('dst_bit_xor')
  plt.subplot(236), plt.axis('off'), plt.imshow(dst_bit_not, 'gray'), plt.title('dst_bit_not_src1')
  plt.show()
  ```

  
