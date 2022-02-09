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

      - 장점

        

      - 단점

