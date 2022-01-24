# 21.12.30 교육

## 1. 결측치와 이상치

- 결측치 : 잘 못 들어온 값, 누락 값(NA로 표현)

  - 삭제 또는 대치

  ```python
  # df1의 a 컬럼 결측치를 a 컬럼의 최소값으로 대치, 그 후 전체 평균 계산
  pd.read_csv('./code/file1.txt')
  df1=pd.read_csv('./code/file1.txt')
  
  df1['a'][df1['a'] == '.'] = np.nan
  
  df1['a'] = df1['a'].astype('float')
  vmin = df1['a'].min()
  df1['a'][df1['a'].isnull()] = vmin
  
  df1['a'].mean()
  ```

  

- 이상치 : 일반적인 범위를 벗어난 데이터

  - 삭제 또는 대치

  - Box plot으로 확인 후 탐색기법

  ```python
  # df1의 d 컬럼 중 16이상 값을 이상치로 판단, 이상치를 16미만 값중 최대값으로 대치한 후 평균 계산
  df1.d[df1.d >= 16]
  vmax = df1.d[df1.d < 16].max()
  df1.d[df1.d >= 16] = vmax
  df1.d.mean()
  ```



---

## 2. scailing

- 변수 스케일링(표준화)

- 설명변수의 서로 다른 범위에 있는 것을 동일한 범주 내 비교하기 위한 작업

  1) **Standard Scailing**

     평균을 0, 표준편차 1로 맞추는 작업

     표준 정규 분포

  2. **MinMax Scailing**

     최소값 0, 최대값 1로 맞추는 작업

  3. **Robust Scailing**

     중앙값 0, IQR 1로 맞추는 작업

  ```python
  # scailing module 불러오기
  from sklearn.preprocessing import StandardScaler as standard
  from sklearn.preprocessing import MinMaxScaler as minmax
  ```

  ```python
  # iris data loading
  from sklearn.datasets import load_iris
  iris_x = load_iris()['data']
  iris_y = load_iris()['target']
  ```

  ```python
  # 1) standard scailing (표준화) : (x-xbar)/sigma
  
  # 서로다른 행끼리 계산
  (iris_x - iris_x.mean(axis=0)) / iris_x.std(axis=0) 
  
  df1 = (iris_x - iris_x.mean(axis=0)) / iris_x.std(axis=0)
  
  # 함수 사용
  m_sc = standard()
  m_sc.fit(iris_x)   # fit : 데이터를 모델에 적합하게 해주는 함수
  m_sc.transform(iris_x)
  ```

  ```python
  # 2) min max scailing (x-x.min())/(x.max()-x.min())
  
  (iris_x - iris_x.min(0)) / (iris_x.max(0) - iris_x.min(0))
  
  df2 = (iris_x - iris_x.min(0)) / (iris_x.max(0) - iris_x.min(0))
  
  df2.max()
  df2.min()
  # max=1, min=0 사이 값이 나오게 됨
  
  # 함수 사용
  m_sc1 = minmax()
  m_sc1.fit(iris_x)    # MinMaxScaler() 사용됨
  df2 = m_sc1.transform(iris_x)
  
  df2.min()
  df2.max()
  ```
  
  ```python
  # train/test 로 분리되어진 데이터를 표준화
  from sklearn.model_selection import train_test_split
  
  train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y)
  
  # 1) train_x, test_x 동일한 기준으로 스케일링
  m_sc1 = minmax()
  m_sc1.fit(train_x)  # train data set 으로만(test set 안건드림)
  
  train_m_sc1 = m_sc1.transform(train_x)
  test_m_sc1 = m_sc1.transform(test_x)
  
  # 훈련용 데이터에 적용
  train_m_sc1.min(0)
  train_m_sc1.max(0)
  
  # 검증용(테스트)데이터에 적용
  test_m_sc1.min(0)  # 0이 아님
  test_m_sc1.max(0)  # 1이 아님
  ```
  
  ```python
  # 서로 다른 기준으로 스케일링(bad model)
  m_sc2 = minmax()
  m_sc3 = minmax()
  
  m_sc2.fit(train_x)   # train data 도 fit 시킴
  m_sc3.fit(test_x)    # test data 도 fit 시킴
  
  train_m_sc2 = m_sc2.transform(train_x)
  test_m_sc2 = m_sc3.transform(test_x)
  
  train_m_sc2.min()
  train_m_sc2.max()
  
  test_m_sc2.min()
  test_m_sc2.max()
  ```
  
  - **scailing 시각화**
  
    ```python
    # 1) figure, subplot 생성
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3)
    
    # 2) 원본 data의 산점도
    import mglearn
    
    ax[0].scatter(train_x[:,0], train_x[:,1], c=mglearn.cm2(0), label='train')
    ax[0].scatter(test_x[:,0], test_x[:,1], c=mglearn.cm2(1), label='test')
    ax[0].legend()
    ax[0].set_title('raw data')
    
    # 3) 올바른 스케일링 data의 산점도(train_x_sc1, test_x_sc1)
    ax[1].scatter(train_m_sc1[:,0], train_m_sc1[:,1], c=mglearn.cm2(0), label='train')
    ax[1].scatter(test_m_sc1[:,0], test_m_sc1[:,1], c=mglearn.cm2(1), label='test')
    ax[1].legend()
    ax[1].set_title('good scaing data')
    
    
    # 4) 잘못된 스케일링 data의 산점도(train_x_sc2, test_x_sc2)
    
    ax[2].scatter(train_m_sc2[:,0], train_m_sc2[:,1], c=mglearn.cm2(0), label='train')
    ax[2].scatter(test_m_sc2[:,0], test_m_sc2[:,1], c=mglearn.cm2(1), label='test')
    ax[2].legend()
    ax[2].set_title('bad scaling data')
    ```
  
    
  
  