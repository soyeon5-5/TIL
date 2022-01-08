# 22.01.07

## 1. np.where

```python
df1 = pd.read_csv('./code/ex_test1.csv', encoding = 'cp949')
# col : 회원번호, 가입일, 첫구매일, 최종구매일, 주문금액

# point 적립 : 주문금액 5만 미만 1%, 10만 미만 2%, 그 이상은 3%

df1['point']=np.where(df1['주문금액']<50000, # 첫번째 조건
                       df1['주문금액']*0.01,   # 첫번째 조건 참이면 연산
                       np.where(df1['주문금액']<100000,   # 두번째 조건
                                df1['주문금액']*0.02,      # 두번째 조건이 참이면 연산
                                df1['주문금액']*0.03))
# 회원 번호 별 총 주문금액과 포인트 확인
df1.groupby('회원번호')[['주문금액','point']].sum()
```



---

## 2. 행렬 변환

#### 1) 역함수와 역변환

```python
import numpy as np

A = np.matrix([[1,0,0,0],[2,1,0,0],[3,0,1,0],[4,0,0,1]])

# 역행렬
print(A)
print(np.linalg.inv(A))

# 전치 행렬(Transpose)
a = np.arange(15).reshape(3,5)
np.transpose(a)
```

#### 2) 고유값, 고유벡터

```python
w, v = np.linalg.eig(a)

print(w)  # 고유값
print(v)   # 고유벡터(단위벡터로 정규화)


b = np.array([[1,3],[4,2]])
np.linalg.eig(b)
w,v = np.linalg.eig(b)

print(w)
print(v)
```



---

- **numpy 라이브러리를 활용한 그래프 그리기**

  ```python
  # 버블차트
  import matplotlib.pyplot as plt
  import random
  
  x = []
  y = []
  size = []
  
  for i in range(200) : 
      x. append(random.randint(10,100)) # x에 10과 100 사이 랜덤한 정수(int)추가
      y. append(random.randint(10,100))
      size.append(random.randint(10,100))
      
  plt.scatter(x, y, s=size, c=x, cmap='jet', alpha=0.7)
  plt.colorbar()
  plt.show()
  
  ```

  
