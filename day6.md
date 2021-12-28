# 21.12.28 교육

# 1. drop, shift, rename

### 1. **drop**

> 특정 행, 컬럼 제거 / 이름 전달

```python
emp = pd.read_csv("./code/emp.csv")

# scott 포함된 컬럼 제거
emp['ename'] == 'scott'  # 조건문
emp.loc[emp['ename'] == 'scott']  # scott 관련된 record 
emp.loc[~(emp['ename'] == 'scott'), :]

# sal 컬럼 제외
emp.drop(4, axis=0)  # 행기준, [4] idx 제외

# emp 데이터셋에서 sal 컬럼 제외
emp.drop('sal', axis=1)
```

### 2. **shift**

> 행/열 을 이동, 증감율에 적용

 ```python
 emp['sal'].shift() # default : axis=0 (행)
 
 #다음 데이터프레임에서 전일자 대비 증감율 출력
 s1 = Series([3000,3500,4200,2800], index = ['2021/01/01','2021/01/02','2021/01/03','2021/01/04'])
 
 ((s1-s1.shift())/s1.shift())*100
 ```

### 3. **rename**

> 행, 컬럼명 변경

 ```python
 emp.columns = ['emptno','ename','deptno','salary']
 emp.rename({'salary':'sal','deptno':'dept_no'}, axis=1)
 ```



---



# 2. stack, unstack, pivot_table

> 데이터타입 형태,  각 속성을 컬럼으로 표현

### 1. **stack**

> wide data -> long data

```python
# kimchi 데이터를 연도별, 제품별 수량의 총합
kimchi = pd.read_csv("./code/kimchi_test.csv", encoding = 'cp949')

# kimchi 데이터를 연도별, 제품별 수량의 총합
df1 = kimchi.groupby(['판매년도','제품'])['수량'].sum()
```

