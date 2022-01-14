# 22.01.14

## 1. 유튜브 카테고리 비율 시각화

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

- **데이터 가져오기**

  ```python
  df = pd.read_excel('./files/youtube_rank.xlsx')[1:101]
  
  # str -> int 변경
  df['replaced_subscriber'] = df['subscriber'].str.replace('만', '0000')
  df['replaced_subscriber'] = df['replaced_subscriber'].astype('int')
  ```

- **category**별 총 합 **pivot_table** 만들기

  ```python
  pivot_df = df.pivot_table(values = 'replaced_subscriber',
                index = 'category',
                aggfunc = ['sum', 'count'])
  
  # index와 column 정리
  pivot_df.columns = ['subscriber_sum', 'subscriber_cnt']
  pivot_df = pivot_df.reset_index()
  
  # sum 값으로 정렬
  pivot_df = pivot_df.sort_values(by = 'subscriber_sum', 
                                  ascending = False)
  
  pivot_df = pivot_df.reset_index(drop = True)
  ```

- 한글 폰트

  ```python
  path = 'c:/Windows/Fonts/malgun.ttf'
  font_name = font_manager.FontProperties(fname = path).get_name()
  rc('font', family = font_name)
  ```

- pie chart

  ```python
  plt.figure(figsize = (6,6))
  plt.rcParams['font.size'] = 10
  plt.pie(pivot_df.head(6)['subscriber_sum'],	  # 상위 6개만
          labels = pivot_df.head(6)['category'], 
         autopct = '%.1f%%')   # float 형태로 소수점 1개, %로 표현
  plt.show()
  ```



---

## 2. 관광객 추이(open data)

