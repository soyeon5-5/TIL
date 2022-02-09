# 22.02.08

## Machine Learning & Scikit-Learn

1. **회귀분석(Regression)**

   2. 로지스틱 회귀 분석(logistic regression)

      종속변수가 범주형(categorical)

      분류 모델

      - 회귀식 : 로지스틱 함수 , 0~1 사이값을 가짐

      - 에러 함수의 최소값 구하는 방법

        1. 최대가능도법

        2. **Coordinate distance algorithm**

           Scikit-learn에서 사용하는 방법

           편미분 미사용

        2. Stochastic average gradient descent algorithm

        2. Newton method

        2. BFGS(Broyden-Fletcher-Goldfard-Shanno)

        2. LBFGS(Limited memory BFGS)

2. **분류(Classification)**

   1. kNN(k-Nearest Neighbor) 모델

      머신러닝 모델 중 가장 직관적, 간단한 지도학습 모델

      유사성 척도(거리함수) 기반 분류 : Euclidean distance

      특정 변수로 인한 결과 편차를 줄이고자 scaling -> 데이터 정규화 변환 : Minmax scaling

   2. SVM(Support Vector Machine)

      보편적으로 사용된 분류 머신러닝 모델

      데이터가 2개의 그룹으로 분류될 때

      주어진 데이터 기반, 그 경계(초평면)를 구하고, 새로운 데이터가 어느 쪽인가 판단하는 모델

      - 구성요소

        Hyper plane(초평면) : 그룹의 경계선, feature보다 한차원 낮음

        Support vector : 경계에 가장 가까운 각 클래스의 데이터

        Margin : 초평면과 서포트벡터의 수직거리

      - 목표

        

   