# 22.01.27

## OpenCV

#### 1. OTSU

> Bimodal에서 Threshold 결정 방법

- 임계값 T 기준, 2개의 그룹으로 나눴을 때, 각 집합내 명암 분포(σw)는 균일하고 집합 간의 명암(σb) 차이가 최대화 될 수 있는 T



#### 2. Morphology(형태학)

> 생물학의 한 분야, 동식물의 모양이나 구조를 다루는 학문

- **Mathematical morphology**(수학적 형태학)

  객체 검출을 원활하게 하기 위해 영상 분할 결과를 단순화 하는 방법으로 사용

  - 객체 경계 단순화, 작은 구멍 채움, 작은 돌기 제거 등

  - Binary, Gray-scale 영상에 적용 가능

  - 모폴로지 필터링

    1. 구조적 요소(structuring element)
    2. 팽창(dilation)
    3. 침식(erosion)

  - **팽창 연산**(Dilation operation)

    작은 구멍 채움, 인접한 두 객체 연결

    구조적 요소의 중심(anchor point)가 영상의 1에 위치하면 연산(구조적 요소 부분을 1로)

  - **침식 연산**(Erosion operation)

    객체 경계 침식, 작은 돌기 제거

    구조적 요소 중 하나라도 영상의 0에 위치하면 연산(anchor 부분 0으로)

  - **열림 연산**(Opening operation)

    침식 후 팽창 연산

    작은 크기의 객체에 포함되는 픽셀 제거

  - **닫힘 연산**(Closing operation)

    팽창 후 침식 연산

    객체 내부의 작은 구멍(hole)이나 간격(gap) 채움

#### 3. Geometric Transforms

> 수식이나 변환 관계에 의해 픽셀들의 위치를 변경하는 변환

1. **spatial transform**(기하학적 변환)

   - Affine transform(linear transform)

     휘어짐 없이 평행한 선들은 그대로 평행을 유지하는 변환

     이동, 회전, 스케일

   - Warping(Nonlinear transform)

     pixel 별로 이동 정도를 다르게 하여 영상을 임의대로 구부린 효과를 낼 수 있음

     고차항 이용

2. **interpolation**(공간 변환, 보간)

   결과 픽셀에 정확히 대응되는 입력 픽셀이 없을 때, 주변 픽셀들을 고려하여 새로운 값을 생성하는 방법

   - Nearest neighbor interpolation(최근접 보간)

     가장 가까운 원시 픽셀값으로 선택

     선택은 빠르나 질이 좋지 않음

   - Neighbor aberaging interpolation(근접 평균 보간)

     근처 값(4개)의 평균 값으로 선택

     연산량 증가, Nearst보다 조금 더 질이 좋아짐

   - Bilinear interpolation(성형 보간)

     근처 값(4개) 픽셀들에 가까운 정도로 가중치를 부여한 값들의 합으로 선택

     연산 속도와 영상의 질이 괜찮음

   - Higher order interpolation(고차항 보간)

     고차항 형태로 계산되어진 값

     시간이 오래 걸림

     b-spline interpolation

     