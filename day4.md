# 21.12.24 교육
## pandas 정렬 sort()

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

pwd
### 현재 위치(present woriking directory)

 pd.read_csv('emp.csv')
# 현재 폴더 열기
` pd.read_csv('./code/emp.csv')`

- 현재 폴더의 하위 폴더인 code의 emp.csv 읽어줘 

`pd.read_csv("C:/Users/username/code/emp.csv") `

- 절대주소로 해도됨

1. **os.getcwd()**

   get current working directory 현재 작업폴더를 가져와라

2. **sort() 정렬**

   sort_index

   Series, DataFrame 호출 가능

3. **index, column 재배치**

   ```python
   emp = pd.read_csv('./code/emp.csv')
   
   emp.ename
   emp['ename']
   emp['empno']
   emp.idx = emp['empno']
   emp.idx
   emp.iloc[:, 1:]
   
   emp = emp.set_index('ename')
   emp.sort_index(ascending=True)  
   # index ename 기준 오름차순, ascending=True 생략 가능
   emp.sort_index(ascending=False)
   # 내림차순
   emp.sort_index(axis=0) # index의 오름차순
   emp.sort_index(axis=1) # column의 오름차순
   
   ```

> ename을 기준으로 정렬, ename이 index가 됨

4. **sort_values**

   Series, DataFrame 호출 가능

   본문 값(value)으로 정렬

   ```python
   emp.sort_values(by = 'sal')
   # sal 기준 오름차순 정렬
   # by = 은 생략 가능
   emp.sort_values('sal', ascending=False)
   # sal 기준 내림차순 정렬
   
   emp.sort_values(['deptno','sal'])
   # deptno 기준 먼저 배열 후 sal 기준 배열
   emp.sort_values(['deptno','sal'], ascending = [True, False])
   ```

   
