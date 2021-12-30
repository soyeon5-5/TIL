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

  

