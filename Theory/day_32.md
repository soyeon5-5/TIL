# 22.02.09

## Machine Learning & Scikit-Learn

2. **분류(Classification)**

   4. Naive Bayes(나이브 베이지안 분류)

      조건부 확률 기반 - 베이지안 룰

      분류 기준 : MAP(Maximum A Posteriori) - 사후 확률을 계산하여 더 높은 확률을 가지는 것을 정답으로 분류

      - Scikit learn에서 제공하는 나이브 베이지안 종류

        가우시안 나이브 베이즈  GaussianNB()

        ​	연속적인 값

        다항분포 나이브 베이즈 MultimoniaNB()

        ​	이산적인 값, 특성이 여러 종류

        베르누이 나이브 베이즈(이항분포) BernoulliNB()

        ​	이산적인 값, 특성이 2종류

   5. Decision Tree(결정 트리)

      어떤 모양의 데이터든 계산 가능

      IG(Information gain) 이 큰 방향, 영역의 순도(homogeneity)증가, 불순도(impurity) 감소 하는 방향으로 분할

      IG = E(Parent) - E(Parent|Motivation)

      - 장점

        시각화될 수 있어 이해하기 쉽고 해석이 간단

        전처리 절차가 간단

        데이터 증가에 따른 트리의 횟수는 log로 증가

        여러 결과 값이 나오는 분석 가능

        White-box 모델

        통계학적 추정을 통한 모델의 신뢰성 평가 가능

      - 단점

        복잡한 트리가 생성되어 데이터 일반화를 잘 못함

        불안정 - 약간의 데이터 변화에도 다른 결정트리 형성될 수 있음

        지역 최적화 방법

        XOR과 같은 문제 표현 어려움

      - 순도 계산 방법

        1.Entropy - classification 모델

        2.Gini index - classification 모델

        3.MSE(Mean square error) - regression 모델