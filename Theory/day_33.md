# 22.02.10

## Machine Learning & Scikit-Learn

### Unsupervised learning(비지도 학습)

1. **Clustering**

   > 비슷한 정도에 따라 클러스터 구성

   클러스터 생성 기준

   1. Centroid model

      기준 데이터를 사용하여 분포에 따라 cluster 구성

      * **K-means** algorithms

        K개의 기준값을 사용해 K개의 cluster 생성

        기준값 조정 규칙 : 클러스터 내 기준값과 데이터들의 거리합 최소, 클러스터 간 거리 최대

   2. Hierachical(Connectivity) model

      서로 관련있는 데이터들을 계층화하여 cluster 구성

      1. **Agglomerative**

         bottom-up 형식

         각 클러스터 간 거리 기준 통합

         - Nearest Neighbor (single)

           클러스터간 거리들 중 가장 가까운 값에 따라 계층 생성 후 가장 가까운 값으로 통합

         - Furthest Neighbor (complete)

           클러스터간 거리들 중 가장 가까운 값으로 계층 생성 후 가장 먼 값으로 통합

         - 거리의 기준들
           1. Centroid : 각 클러스터 내 모든 원소들과 거리가 동일한 중앙값들 간의 거리를 기준
           2. Average : 각 클러스터 내 모든 원소간의 거리의 평균을 기준
           3. Median : 각 클러스터 내 모든 원소간의 거리의 중앙값을 기준
           4. Ward : 각 클러스터 내 모든 원소간의 평균을 기준이되 공분산을 고려한 보정 진행

      2. **Divisible**

         top-down 형식

   3. Density model

      데이터들이 밀집된 구역끼리 cluster 구성

   4. Distribution model

      확률 기반 모델, 기존 확률분포모델 중 데이터에 적합한 모델을 이용하여 cluster 구성

      데이터 분포에 따른 cluster 구성이라 해당 데이터에 대해 overfitting 될 수 있음

   

   
