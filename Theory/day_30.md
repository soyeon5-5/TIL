# 22.02.07

## Machine Learning & Scikit-Learn

### Scikit-Learn

> 파이썬으로 구현된 라이브러리 중 머신 러닝 교육 및 실무용으로 가장 많이 사용되는 open source library

- **Bunch 클래스**

  - 속성

    data (필수) : 독립변수 ndarray 배열 

    target (필수) : 종속변수 ndarray 배열 

    feature_names (옵션) : 독립 변수 이름 리스트

    target_names (옵션) : 종속 변수 이름 리스트 

    DESCR (옵션) : 자료에 대한 설명 

### Data preprocessing

> 현실에서 가져오는 raw data의 품질 보완 작업

- **데이터 정제(Cleaning, Cleansing)**

  결측값(missing value)을 채우거나, 잡음값(noisy data)을 평활화(smoothing), 이상치(outlier)  제거, 불일치 해결

- **데이터 통합(Intergration)**

  다수의 소스에서 얻은 데이터 통합

- **축소(Reduction)**

  크기는 작으나 분석 결과 동일한 데이터로 표현

- **변환(Transformation)**

  기계학습 알고리즘의 효율성을 극대화하기 위한 변형

- **표준화(Standardization = mean removal and variance scaling)**

  선형 변환을 적용하여 전체 자료를 평균 0, 분산 1이 되도록 만드는 과정, overflow나 underflow를 방지하기 위함

  - 관련 함수

    scale(X): 표준정규분포 스케일

    robust_scale(X): meadian 사용. Outlier의 영향 최소화

    minmax_scale(X): 최대/최소값 사용

    maxabs_scale(X): 최대 절대값 사용

- **정규화(Normalization = scaling and centering)**

  개별 데이터의 크기를 모두 동일하게 만드는 변환

  다차원의 독립변수 벡터가 있을 때, 각 벡터 원소들의 상대적인 크기만 중요한 경우 사용

  - 관련 함수

    normalize(X)

### 인공지능

> 기계가 학습과 추리 등의 인간의 지능과 비슷한 작업을 수행하는 것

**인공지능⊃머신러닝(기계학습)⊃딥러닝**

- 기계학습

  데이터에 대한 수학적 모델 기반 인공지능 기법

- 딥러닝

  Artificial Neural Network 기반 인공지능 기법

- 분류

  약한 인공지능 : 인지 모델링 시스템

  -인간처럼 생각하는 시스템, 인간처럼 행동하는 시스템

  강한 인공지능 : 계산 모델 시스템

  -합리적으로 생각하는 시스템, 합리적으로 행동하는 시스템

- **기계학습의 범주**

  1. 지도학습(Supervised Learning)

     (특징,결과)로 주어진 훈련 데이터로 학습

     분야 : 분류, 회귀

  2. 비지도학습(Unsupervised Learning)

     (특징)만 주어짐

     분야 : 군집화, 차원축소

  3. 강화학습(Reinforment Learning)

- **기계 학습의 적용 분야**

  1. 회귀(Regression)

     학습 자료에 대한 근사식을 구해 새로운 자료에 대한 레이블 예측

     방법론 : Linear regression, Logistic regression, Polynomial regression, etc.

  2. 분류(Classification)

     둘 이상의 이산적인 범주로 레이블 예측

     방법론 : Naive Bayes, Decision Tree, Logistic Regression, K-Nearest Neighbors, SCM(Suppor Vector Machine)

  3. 군집화(Clustering)

     레이블이 없는 데이터에 대한 레이블 추론

     방법론 : K-means clustering, DBSCAN, etc
     
  4. 차원 축소
  
     많은 feature로 구성된 다차원 데이터의 차원을 축소하여 새로운 차원의 데이터 생성
  
     방법론 : PCA(Principal Component Analysis), LSA(Latent Semantic Analysis)

### 머신러닝과 통계학

새로운 입력값에 대한 결과를 예측하는 과정에서 확률 이론 활용

- 기계학습에서 통계

  상관관계분석

  회귀분석

  확률분석
  
- 상관분석(correlation analysis)

  - 상관계수(correlation coeffieint), r

    범위 : -1≤r≤1

    수식 계산 혹은

    Pearson's correlation 사용 - Pandas의 corr() 함수 사용

    





