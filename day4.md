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

   1. sort_index

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
   
   emp.set_index('ename')
   ```

> ename을 기준으로 정렬, ename이 index가 됨

​		
