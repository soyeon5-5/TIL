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
  ```

  

