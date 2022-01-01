# 21.12.29 교육

## 1. 날짜 표현

- 현재 날짜

  ```python
  import numpy as np
  import pandas as pd
  from pandas import Series, DataFrame
  
  from datetime import datetime
  datetime.now()
  
  d1 = datetime.now()
  d1.year     # 연
  d1.month    # 월
  d1.day      # 일
  
  ```

- 날짜 포맷 변경

  날짜 포맷 변경 후 return data type은 문자

  ```python
  d1 = datetime.now()
  datetime.strftime(d1, '%A')  #완전체
  datetime.strftime(d1, '%a')  #축약형
  # 요일 리턴  Wednesday
  datetime.strftime(d1, '%m-%d,%Y')
  # 12-29,2021
  ```

- 날짜 연산

  ```python
  d1      # 현재 날짜
  # 1) offset
  from pandas.tseries.offsets import Day, Hour, Second
  d1+Day(100)
  #Timestamp('2022-04-08 11:43:03.597602')
  
  # 2) timedelta (날짜와의 차이)
  from datetime import timedelta
  d1+timedelta(100)
  #datetime.datetime(2022, 4, 8, 11, 43, 3, 597602)  d1형식 유지
  
  # 3) DataOffset 
  d1 + pd.DateOffset(month = 4)
  
  # 4) 날짜 - 날짜
  d2='2022/01/01'
  d3 = d1 - datetime.strptime(d2, '%Y/%m/%d')
  d3.days
  d3.seconds
  ```



---

## 2. 데이터 시각화

- 선 그래프

  ```python
  import numpy as np
  import pandas as pd
  from pandas import Series, DataFrame
  
  import matplotlib.pyplot as plt
  
  
  s1.plot(xticks=[0,1,2,3],  # 눈금 좌표
          ylim=[0,100],      # y축 범위
          xlabel='x name',   # x축 라벨
          ylabel='y name',   # y 축 라벨
          rot=90,            # rot= rotation 회전 90도
          title='name',      # title
          marker='^',        # 마커
          linestyle='--',    # 선 스타일
          color='red')       # 선 색상
          
  
  ```

  

