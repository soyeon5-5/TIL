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

## 2. 관광객 현황(open data)

```python
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import rc, font_manager
import matplotlib.pyplot as plt

path = 'c:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = path).get_name()
rc('font', family = font_name)
```

- 데이터 불러오기

  ```python
  kto_201901 = pd.read_excel('./files/kto_201901.xlsx',
                            header = 1,
                            usecols = 'A:G',
                            skipfooter = 4)
  kto_201901.head()
  
  kto_201901.info() 
  # 국적만 object, 다른 column 모두 int
  ```

- 대륙 데이터 값 추가

  ```python
  kto_201901['기준년월'] = '2019-01'
  
  continents_list = ['아시아주', '미주', '구주', '대양주', '아프리카주', '기타대륙', '교포소계']
  condition = kto_201901['국적'].isin(continents_list)
  
  #kto_201901_country = kto_201901[~condition]도 가능
  kto_201901_country = kto_201901[condition == False]
  kto_201901_country_newindex = kto_201901_country.reset_index(drop = True)
  
  continents = ['아시아']*25 + ['아메리카']*5 + ['유럽']*23 + ['오세아니아']*3 \
  + ['아프리카']*2 + ['기타대륙'] + ['교포']
  
  kto_201901_country_newindex['대륙'] = continents
  ```

- 관광객 비율

  ```python
  kto_201901_country_newindex['관광객비율(%)'] = \
  round(kto_201901_country_newindex['관광'] * 100 / kto_201901_country_newindex['계'], 2)
  
  kto_201901_country_newindex.sort_values(by = '관광객비율(%)',
                                         ascending = False)
  
  kto_201901_country_newindex.pivot_table(values = '관광객비율(%)',
                              index = '대륙')
  
  
  # 국적별 비율 확인(예시)
  condition = (kto_201901_country_newindex['국적'] == '중국')
  kto_201901_country_newindex[condition]
  ```

- 국적별 비율 추가

  ```python
  # 국적별 비율 확인(예시)
  condition = (kto_201901_country_newindex['국적'] == '중국')
  kto_201901_country_newindex[condition]
  
  # 비율 추가
  tourist_sum = sum(kto_201901_country_newindex['관광'])
  kto_201901_country_newindex['전체비율(%)'] = \
      round(kto_201901_country_newindex['관광'] / tourist_sum * 100, 1)
  
  kto_201901_country_newindex.sort_values(by = '전체비율(%)', ascending = False)
  ```

- **함수**로 만들기

  ```python
  def create_kto_data(yy, mm):  
      #1. 불러올 Excel 파일 경로를 지정
      file_path = './files/kto_{}{}.xlsx'.format(yy, mm)  
      
      # 2. Excel 파일 불러오기 
      df = pd.read_excel(file_path, header=1, skipfooter=4, usecols='A:G')
      
      # 3. "기준년월" 컬럼 추가 
      df['기준년월'] = '{}-{}'.format(yy, mm) 
      
      # 4. "국적" 컬럼에서 대륙 제거하고 국가만 남기기 
      ignore_list = ['아시아주', '미주', '구주', '대양주',
                     '아프리카주', '기타대륙', '교포소계']    # 제거할 대륙명 선정하기 
      condition = (df['국적'].isin(ignore_list) == False)# 대륙 미포함 조건 
      df_country = df[condition].reset_index(drop=True) 
      
      # 5. "대륙" 컬럼 추가 
      continents = ['아시아']*25 + ['아메리카']*5 + ['유럽']*23 + ['대양주']*3 + ['아프리카']*2 + ['기타대륙'] + ['교포']    # 대륙 컬럼 생성을 위한 목록 만들어 놓기 
      df_country['대륙'] = continents   
  #     df_country = pd.concat([df_country, pd.Series(continents, name = '대륙')], axis=1)                   
      
      # 6. 국가별 "관광객비율(%)" 컬럼 추가
      df_country['관광객비율(%)'] = round(df_country.관광 / df_country.계 * 100, 1) 
                         
      # 7. "전체비율(%)" 컬럼 추가
      tourist_sum = sum(df_country['관광'])
      df_country['전체비율(%)'] = round(df_country['관광'] / tourist_sum * 100, 1)
      
      # 8. 결과 출력
      return(df_country)
  ```

- **함수 적용**

  ```python
  df = pd.DataFrame()
  for yy in range(2010, 2021) :
      for mm in range(1, 13):
          mm_str = str(mm).zfill(2)
          
          try:
              temp = create_kto_data(str(yy), mm_str)
              df = df.append(temp, ignore_index = True)
          except :
              pass
              
  df.to_excel('./files/kto_total.xlsx', index = False)
  ```

- **국적별 데이터 저장**

  ```python
  country_list = df['국적'].unique()
  # country_list = set(df['국적'])
  
  for cntry in country_list :
      condition = df['국적'] == cntry
      df_filter = df[condition]
  
      file_path = './files/[국적별 관광객 데이터] {}.xlsx'.format(cntry)
      df_filter.to_excel(file_path, index = False)
  ```

- 저장된 데이터로 **그래프** 그리기

  ```python
  df = pd.read_excel('./files/kto_total.xlsx')
  
  # 중국인 관광객 그래프
  df_filter = df[df['국적'] == '중국']
  plt.figure(figsize = (12,6))
  plt.plot(df_filter['기준년월'], df_filter['관광'])
  plt.title('중국 관광객 추이')
  plt.xlabel('기준년월')
  plt.ylabel('관광객 수')
  plt.xticks(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01', '2015-01', '2016-01', '2017-01', '2018-01', '2019-01', '2020-01'])
  
  plt.show()
  ```

  
