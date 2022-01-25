# 22.01.24

## 컴퓨터 비전(Computer Vision)

#### 1. 2차원 -> 디지털화

> Sampling - Quantizing - Coding

- **Sampling**

  표본화

  1. Picture Element(pixel, pel) : 화소로 표현

     pixel은 왼쪽 위가 0,0

     I(0,0) 으로 표현, I는 intensity

     Pixel per Inch(PPI)

  2. Dpi(dots per inch)

  2. Volume Element(Voxel)

     MIR, CT

  좌표계

  1. Cartesian coordinate -직교좌표계(x, y)
  2. Polar coordinate - 극좌표계(r, θ)

- **Quantization**

  양자화

- **Coding**

  부호화

  저장방법

  1. jpg : 손실압축
  2. png : 무손실압축
  3. bmp : 무손실 



---

#### 2. 디지털 영상

- **표현 방법**

  영상 좌표(x, y) : OpenCV

  행렬 좌표(y, x) : np.array

- **유형(mode)**

  1. binary image

     0과 1로만 표현(흑 백)

     text, mask image

     Dithering : 점의 밀도로 표현, 눈의 착시 이용

     Halftoning : 공간적 통합작용 이용, 연속적인 이미지를 패턴 혹은 점으로 표현하는 과정, 프린팅에서 사용

  2. grayscale image

     0~255 까지의 8bit 로 표현

  3. color image

     24bit로 표현  - RGB로 8bit가 3개

     - True color image

       한 픽셀에 각각의 RGB값이 나오면서 색상 표현

       OpenCV : BGR

       Matplotlib : RGB

     - Indexed color image 

       한 픽셀에서 RGB값이 동시에 나오면서 색상표현

       gif 파일, 영상처리 X
  
  4. multi-spectral image
  
     적외선 사진 + 위성 사진 과 같이 multi image
