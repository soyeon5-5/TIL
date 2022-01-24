# 21.12.27 교육

# 1. python pandas groupby

> 그룹연산

```python
import pandas as pd
from pandas import Series, DataFrame

#pd.read_csv("절대경로",인코딩)
kimchi = pd.read_csv("./kimchi_test.csv", encoding = 'cp949')
#kimchi.groupby(by=None, # 그룹핑 할 컬럼(기준)
#               axis=0,  # 그룹핑 연산 방향
#               level: None) # 멀티 인덱스일 경우 특정 레벨의 값을 그룹핑 컬럼으로 사용

kimchi.groupby(by=['제품','판매처'])['수량'].sum()
# 제품 기준, 수량과 판매금액 총 합 구하기

kimchi.groupby(by=['제품'])[['수량','판매금액']].sum()
# 이중리스트
```

- **agg**

  ```python
  kimchi.groupby(by=['제품','판매처'])[['수량','판매금액']].agg(['sum','mean'])
  # 제품별, 판매처별(김치별) 수량 판매금액 총합,평균
  
  kimchi.groupby(by=['제품','판매처'])[['수량','판매금액']].agg({'수량':'sum', '판매금액':'mean'})
  #제품별, 판매처별(김치별) 수량은 총합만, 판매금액은 평균
  ```

- **multi index**

  ```python
  kimchi_2 = kimchi.groupby(by=['제품','판매처'])['수량'].sum()
  kimchi_2.groupby(level=0).sum() # 제품별 총합
  kimchi_2.groupby(level='제품').sum()
  kimchi_2.groupby(level=1).sum() # 판매처별 총합
  ```



---



# 2. concat vs merge

> 연관된 데이터 병합

### 1. **concat**

```python
df1 = DataFrame(np.arange(1,7).reshape(2,3), columns=['A','B','C'])
df2 = DataFrame(np.arange(10,61,10).reshape(2,3), columns=['A','B','C'])

pd.concat([df1,df2],axis=0)
# 기본 설정 - 서로다른 행끼리 결합

pd.concat([df1,df2],axis=1)
# 서로다른 열끼리 결합

pd.concat([df1,df2], ignore_index=True)
# 기존 인덱스 무시, 순차적 인덱스 부여
```



### **2. merge**

​	등가 조건만을 사용하여 조인

```python
''' 
pd.merge(left,          # 첫번째 데이터프레임
         right,         # 두번째 데이터프레임
         how='inner',   # 조인방법(default =  'inner')
         on=,           # 조인하는 기준 컬럼(컬럼명이 같을 때)
         left_on=,      # 컬럼명이 다를 때-첫번째 데이터 프레임 조인
         right_on=)     # 컬럼명이 다를 때-두번째 데이터 프레임 조인
'''

df_dept = DataFrame({'deptno':[10,20,30],'dname':['인사부','총부무','IT분석팀']})

pd.merge(emp, df_dept, on='deptno')
#default - inner 조인

pd.merge(emp, df_dept, how="left",on='deptno')
# outer join

DataFrame({'deptno':[10,20],'dname':['인사총무부','IT분석팀']})

```

