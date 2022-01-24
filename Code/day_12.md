# 22.01.06 교육

## 1. 차원 축소

> 분석 대상의 변수를 최대한 유지하며 변수 개수를 줄이는 탐색적 분석기법
>
> 다른 분석 전단계,  분석 후 개선 방법, 시각화 목적
>
> 저차원일 경우 머신러닝 작동 원활

### 1) PCA

```python
# data loading
from sklearn.datasets import load_iris
iris_x = load_iris()['data']
iris_y = load_iris()['target']
```

- **2차원 축소** : 주성분 개수 2개

  ```python
  # PCA 적용 전 스케일링 변환
  from sklearn.preprocessing import StandardScaler as standard
  m_sc = standard()
  m_sc.fit_transform(iris_x)
  iris_x_sc = m_sc.fit_transform(iris_x)
  
  # 2차원으로 축소
  from sklearn.decomposition import PCA
  m_pca2 = PCA(n_components = 2)
  iris_x_pca2 = m_pca2.fit_transform(iris_x_sc)
  ```

- **인공변수로 시각화** (컬럼 선별)

  ```python
  import mglearn
  mglearn.discrete_scatter(iris_x_pca2[:,0], iris_x_pca2[:,1], y = iris_y)
  ```

- **3차원  축소**

  ```python
  from sklearn.decomposition import PCA
  m_pca3 = PCA(n_components=3)
  iris_x_pca3 = m_pca3.fit_transform(iris_x_sc)
  ```

- **도화지 그리기, 축 그리기**

  ```python
  fig1 = plt.figure()   # 도화지
  ax = Axes3D(fig1)     # 축
  
  
  # step 1. y == 0 인 데이터 포인트만 시각화
  ax.scatter(iris_x_pca3[iris_y==0,0],  # x축 좌표
             iris_x_pca3[iris_y==0,1],  # y축 좌표 
             iris_x_pca3[iris_y==0,2],  # z축 좌표
             c = 'b',
             cmap = mglearn.cm2,
             s = 60,                # 점의 크기
             edgecolors = 'k')      # 블랙 색상
  
  # step 2. y == 1 인 데이터 포인트만 시각화
  ax.scatter(iris_x_pca3[iris_y==1,0],  # x축 좌표
             iris_x_pca3[iris_y==1,1],  # y축 좌표 
             iris_x_pca3[iris_y==1,2],  # z축 좌표
             c = 'r',
             cmap = mglearn.cm2,
             s = 60,                
             edgecolors = 'k')      
  
  
  # step 3. y == 2 인 데이터 포인트만 시각화
  ax.scatter(iris_x_pca3[iris_y==2,0],  # x축 좌표
             iris_x_pca3[iris_y==2,1],  # y축 좌표 
             iris_x_pca3[iris_y==2,2],  # z축 좌표
             c = 'y',
             cmap = mglearn.cm2,
             s = 60,                
             edgecolors = 'k')      
  
  # 모델 적용 (KNN_최근접 이웃)
  from sklearn.neighbors import KNeighborsClassifier as knn
  
  m_knn1 = knn()
  m_knn2 = knn()
  
  from sklearn.model_selection import train_test_split
  
  
  train_x1, test_x1, train_y1, test_y1 = train_test_split(iris_x_pca2, iris_y, random_state=0)
  train_x2, test_x2, train_y2, test_y2 = train_test_split(iris_x_pca3, iris_y, random_state=0)
  # random_state = 0 : 초기값 설정 seed 값 고정
  
  m_knn1.fit(train_x1, train_y1)
  m_knn1.score(test_x1, test_y1)
  
  m_pca2.explained_variance_ratio_  # 각 인공변수의 분산 설명력
  sum(m_pca2.explained_variance_ratio_)
  
  
  m_knn2.fit(train_x2, train_y2)
  m_knn2.score(test_x2, test_y2)
  m_pca3.explained_variance_ratio_
  sum(m_pca3.explained_variance_ratio_)
  ```

---

### 2) MDS

>객체 간 유사성, 비유사성을 거리로 측정하여 공간상 점으로 표현
>
>stress 크기로 차원 축소에 대한 적합도 판단(0: 완벽, 5: 좋음, 10: 보통, 20: 나쁨)

```python
# 1. data loading 
from sklearn.manifold import MDS
from sklearn.datasets import load_iris
iris_x=load_iris()['data']
iris_y=load_iris()['target']

iris_x  # 이 값은 변수가 4개 -->> 4차원
```

- **차원 축소**

  ```python
  # MDS 적용 전 스케일링 변환
  from sklearn.preprocessing import StandardScaler as standard
  m_sc=standard()
  m_sc.fit_transform(iris_x)
  iris_x_sc=m_sc.fit_transform(iris_x)
  
  # 차원 축소
  m_mds2 = MDS(n_components=2)
  m_mds3 = MDS(n_components=3)
  
  iris_x_mds1 = m_mds2.fit_transform(iris_x_sc)
  iris_x_mds2 = m_mds3.fit_transform(iris_x_sc)
  ```

- **유도된 인공변수**

  ```python
  m_mds2.stress_ # --> 적합도 평가(.stress_)
  m_mds2.embedding_  # 변환된 데이터셋 값(embedding_)
  
  # 변환된 데이터 셋 값 ---> points 변수에 담기
  points = m_mds2.embedding_
  ```

- **스트레스 계산**

  ```python
  # 2차원
  import numpy as np
  from sklearn.metrics import euclidean_distances
  
  DE = euclidean_distances(points)
  DA = euclidean_distances(iris_x)   # 실제 거리
  
  stress = 0.5*np.sum((DE-DA)**2)
  stress1 = np.sqrt(stress/(0.5*np.sum(DA**2)))
  
  # 3차원
  m_mds3.stress_
  m_mds3.embedding_
  
  points2 = m_mds3.embedding_ 
  
  DE2 = euclidean_distances(points2)
  DA2 = euclidean_distances(iris_x)
  
  stress2 = 0.5*np.sum((DE2-DA2)**2)
  stress3 = np.sqrt(stress2/(0.5*np.sum(DA2**2)))
  ```

  