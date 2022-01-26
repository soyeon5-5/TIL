# 22.01.25

## Computer Vision

#### 1. 이용되는 곳

1. **공장 자동화 시스템**
   - 품질 관리
   - 하자
2. **생체 시스템**
   - DNA 분석
   - 지문 인식
   - 망막 인식
3. **medical 진단 시스템**
   - Skin cancer diagnosis(피부암 진단)
   - Diabetic retinopaty diagnosis(당뇨성 망막증 진단)
   - Lunit 의 AI chest X-ray(흉부) 암 진단
4. **지능형 교통 시스템**
5. etc

#### 2. Pattern Recognition

> Input object(pattern)를 주어진 algorithm에 의해 category나 class로 classification하는 절차

- 컴퓨터는 같은 물체를 다른 각도, 명도 등으로 구분하여 다르다 판단 -> 이러한 문제를 해결하기 위해 pattern recognition 함

  1. image enhancement(개선)

  2. mage segmentation(분할)
  3. feature extraction(특성추출)
  4. pattern classification : ML(Machine Learning)

#### 3. Image Processing

>컴퓨터를 사용해 기존 영상을 개선, 수정하여 사람이 사용하기 편리하게 하는 분야

- 1. image restoration(복원)

     왜곡이 생긴 원인을 알고 있을 때

  2. image enhancement

     왜곡이 생긴 원인을 모를 때

  3. image compression

     어느 정도 압축해도 시각적으로 차이를 못 느낄 때(데이터 축소, 빠른 진행)

## Point Operation

> pixel 단위로 연산

- 하는 **이유**

  1. image contrast(대조도)

  2. image brightness(밝기)

     -> 이러한 과정이 image enhancement

- **Image histogram**

  x : intensity(0~255)

  y : frequency(해당하는 픽셀 수)

- **Scalar Arithmetic Operation**

  O(x,y) = k * I(x,y) + l

  ​	l : level, k : gain

   * **클리핑(Clipping) 처리**

     if (O(x,y) > 255) O(x,y) = 255;

     if (O(x,y) < 0) O(x,y) = 0;

     - 범위 이상의 값을 범위 내로 조절
     - 곱/나누기 : I값이 작은 어두운 부분은 큰 영향 없이 밝은 부분이 더 밝아짐, contrast(대조도) 증가
     - 덧/뺄셈 : 전체적으로 어두워지거나 밝아짐

- Image Arithmetic Operation

  Absolute difference : 차이를 절댓값으로 나타냄

  Thresholding -  binary image 이용하여 차이 표현