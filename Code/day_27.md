# 22.01.28

## OpenCV

```python
import numpy as np
import cv2
import sys
```



### 1. 영상 히스토그램

```python
import matplotlib.pyplot as plt


# 그레이스케일 영상의 히스토그램
src = cv2.imread('fig/lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()
    
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
hist = cv2.calcHist([src], [0], None, [256], [0, 256])

cv2.imshow('src', src)
plt.plot(hist)
plt.show()

cv2.waitKey()

cv2.destroyAllWindows()


# 컬러 영상의 히스토그램
src = cv2.imread('fig/lenna.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

hist_b = cv2.calcHist([src], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([src], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([src], [2], None, [256], [0, 256])

plt.plot(hist_b, color = "b")
plt.plot(hist_g, color = "g")
plt.plot(hist_r, color = "r")
plt.show()

for 함수 이용
for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(hist, color=c)

cv2.imshow('src', src)
cv2.waitKey()

plt.show()

cv2.destroyAllWindows()
```



### 2. 히스토그램 변환(Histogram modification)

- **명암비 조절(Stretching)**

  ```
  src = cv2.imread('./fig/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)
  
  #normalization, dtype -1은 인풋과 아웃풋을 같게하라
  dst_norm = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX, -1)
  
  #equalization
  dst_equal = cv2.equalizeHist(src)
  
  cv2.imshow('src', src)
  cv2.imshow('dst_norm', dst_norm)
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  ```

- **평활화(Equalization)**

  ```
  src = cv2.imread('./fig/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)
  
  #equalization
  dst_equal = cv2.equalizeHist(src)
  
  cv2.imshow('src', src)
  cv2.imshow('dst_equal', dst_equal)
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  ```

  - **Color processing**

    ```python
    # rgb -> hsv -> rgb
    
    src = cv2.imread('./fig/field.bmp', cv2.IMREAD_COLOR)
    
    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    h, s, v = cv2.split(src_hsv)
    
    v_eq = cv2.equalizeHist(v)
    
    src_hsv_eq = cv2.merge((h, s, v_eq))
    
    src_hsv_eq_bgr = cv2.cvtColor(src_hsv_eq, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('src', src)
    cv2.imshow('src_hsv_eq_bgr', src_hsv_eq_bgr)
    
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    ```

    

  

