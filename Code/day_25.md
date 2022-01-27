# 22. 01.26

## OpenCV

```python
import numpy as np
import cv2
import sys
```



#### 1. Mask 로 알파 채널 이용하기

```python
src = cv2.imread('./fig/puppy.bmp', cv2.IMREAD_COLOR)
# 알파 채널은 unchanged로 읽어오기
img_alpha = cv2.imread('./fig/imgbin_sunglasses_1.png', cv2.IMREAD_UNCHANGED) 

# print(img_alpha.shape)  # (480, 960, 4), 마스크 영상 포함됨

if src is None or img_alpha is None :
    print('image read failed')
    sys.exit()

img_alpha = cv2.resize(img_alpha, (300, 150))
    
sunglass = img_alpha[:, :, 0:3]  # 3번째 까지인 영상만 추출    
mask = img_alpha[:, :, -1] # 마지막 mask만 추출

h, w = mask.shape[:2]
crop = src[120:120+h, 220:220+w]  # glass mask와 같은 크기의 부분 영상 추출

# cv2.copyTo(glass, mask, crop)
crop[mask > 0] = (0, 0, 255)

cv2.imshow('src', src)
cv2.imshow('glass', sunglass)
cv2.imshow('mask', mask)
cv2.imshow('crop', crop)

cv2.waitKey()
cv2.destroyAllWindows()
```



#### 2. 그리기 함수

- 연습

  ```python
  # 빈(흰색) 영상 만들기
  img = np.full((600, 1000, 3), 255, dtype = np.uint8)
  
  # 1. 직선 그리기
  # cv2.line은 image 위에 바로 생성 되므로 원본 유지해야할 경우 반드시 copy하기
  # 직선의 시작점, 끝점(영상좌표 기준)
  # line모양 default는 line_8
  cv2.line(img, (100, 50), (300, 50), (0, 0, 255), 10)
  # 끝 쪽 방향 화살표
  cv2.arrowedLine(img, (400, 20), (400, 280), (0, 0, 255), 10)
  
  # 2. 사각형 그리기
  # 2-1. 좌측 상단, 우측 하단만 입력
  cv2.rectanlge(img,(200, 300), (380, 400), (0, 0, 255), 10)
  # 2-2. 각 사각형 위치(x, y, w, h) 입력
  cv.rectangle(img, (200, 300, 180, 100), (255, 0, 0), -1)
  
  # 3. 원형 그리기
  # 원점, 반지름
  cv2.circle(img, (600, 300), 100, (255, 255, 0), 10, cv2.LINE_AA)
  
  # 4. 타원 그리기
  # 원점, x,y의 반지름, 기울어진 각도(시계방향 기준), 시작, 끝 - 예를 들어 360에 280을 쓰면 덜 그려짐 
  cv2.ellipse(img, (600, 300), (50, 100), 10, 0, 360, (0, 255, 0), 10)
  
  # 5. 글자 넣기
  text = 'Opencv version = ' + cv2.__version__
  cv2.putText(img, text, (700, 100), cv2.FONT_HERSHEY_SIMPLEX,
              0.8, (0, 0, 255), 1, cv2.LINE_AA)
  
  
  cv2.imshow('canvas', img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  ```

- 인삼 쓰기

  ```python
  img = np.full((600, 1000, 3), 255, dtype = np.uint8)
  
  cv2.circle(img, (100, 200), 50, (0, 0, 0), 5, cv2.LINE_AA)
  cv2.line(img, (200, 130), (200, 270), (0, 0, 0), 5)
  cv2.line(img, (120, 300), (120, 360), (0, 0, 0), 5)
  cv2.line(img, (120, 360), (210, 360), (0, 0, 0), 5)
  
  cv2.line(img, (350, 150), (300, 250), (0, 0, 0), 5,  cv2.LINE_AA)
  cv2.line(img, (350, 150), (400, 250), (0, 0, 0), 5,  cv2.LINE_AA)
  cv2.line(img, (450, 130), (450, 270), (0, 0, 0), 5)
  cv2.line(img, (450, 200), (500, 200), (0, 0, 0), 5)
  cv2.rectangle(img, (350, 300), (450, 360), (0, 0, 0), 5)
  
  cv2.imshow('canvas', img)
  
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  ```



#### 3. 카메라와 동영상

```python
# 내 웹캠 영상 불러오기
cap = cv2.VideoCapture(0)  # 0 대신 동영상 파일을 열어도 가능

if not cap.isOpened():
    print('Videocap open failed')
    sys.exit()
    

# w, h, fps 입력, 직접 입력도 가능함
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
#frame per second, 내 컴퓨터 확인
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 비디오 열기 및 저장하기
out = cv2.VideoWriter('ouput.avi', fourcc, fps, (w, h))

while True :
    # T/F가 ret, frame에 영상들어옴
    ret, frame = cap.read()
    
    if not ret :
        print('video read failed')
        break
    
    edge = cv2.Canny(frame, 30, 150)
    
     cv2.imshow('img', frame)
     cv2.imshow('img', edge)

    
    out.write(frame)
    
    # while 돌수 있게 빠르게 돌리기
    # 영상이 들어오는 속도보다 빨라야 안 끊김
    if cv2.waitKey(20) == 27 :
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()
```

- Edge 저장하기

  ```python
  cap = cv2.VideoCapture(0)
  
  if not cap.isOpened():
      print('Videocap open failed')
      sys.exit()
      
  
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  
  
  out_e = cv2.VideoWriter('ouput_edge.avi', fourcc, fps, (w, h))
  
  while True :
      ret, frame = cap.read()
      
      if not ret :
          print('video read failed')
          break
          
     ################################## 
      edge = cv2.Canny(frame, 30, 150)
      edge_inv = 255-edge
      #혹은 edge_inv = ~edge
      edge_color = cv2.cvtColor(edge_inv,cv2.COLOR_GRAY2BGR)
      #컬러형식만 저장이됨
     ################################## 
       cv2.imshow('img', frame)
       cv2.imshow('img', edge_color)
  
      
      out_e.write(edge_color)
      
      if cv2.waitKey(20) == 27 :
          break
          
  cap.release()
  out_e.release()
  cv2.destroyAllWindows()
  ```

  