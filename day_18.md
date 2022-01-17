# 22.01.17

## Instagram Crawling

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import time
```



#### 1. 데이터 crawling 후 저장

```
ser = Service('../chromedriver/chromedriver.exe')
driver = webdriver.Chrome(service = ser)
url = 'https://www.instagram.com/'
driver.get(url)

from selenium.webdriver.common.by import By
```

- 게시글 데이터 **함수**

  ```python
  # 게시글 내용, 해시태그, 게시날짜, 좋아요 수, 장소 함수
  def get_content(driver) :
      html = driver.page_source
      soup = BeautifulSoup(html, 'html.parser')
      
      try : 
          content = soup.select('.C4VMK > span')[0].text
          content = unicodedata.normalize('NFC', content)
      except :
          content = " "
      
      try :
          tags = re.findall(r'#[^\s#,\\]+', content)
      except : 
          tags = " "
          
          
      date = soup.select('time._1o9PC.Nzb55')[0]['datetime'][:10]
      
      try :
          like = soup.select('a.zV_Nj > span')[0].text
      except :
          like = " "
          
      try :
          place = soup.select('div.O4G1U')[0].text
          place = unicodedata.normalize('NFC', place)
      except :
          place = " "
          
      data = [content, date, like, place, tags]
      
      return(data)
  ```

- **다음 게시글**

  ```python
  def move_next(driver) :
      right = driver.find_element(By.CSS_SELECTOR, 'div.l8mY4.feth3')
      right.click()
      time.sleep(3)
      
  move_next(driver)
  ```

- 게시글별 데이터 저장 **함수**

  ```python
  def insta_crawling(word, n):
      url = insta_searching(word)
      
      driver.get(url)
      time.sleep(5)
      
      select_first(driver)
      time.sleep(5)
      
      target = n
      results = []
      
      for i in range(n) :
          try :
              data = get_content(driver)
              results.append(data)
              move_next(driver)
              
          except :
              time.sleep(2)
              move_next(driver)
      
      return(results)
  ```

- 데이터 수집 후 저장

  ```python
  result_1 = insta_crawling('제주도맛집', 3)
  results_df = pd.DataFrame(result_1)
  results_df.columns = ['content', 'date', 'like', 'place', 'tags']
  results_df.to_excel('./files/1_crawling_jeju_3.xlsx', index = False)
  
  # 검색어별 저장 후 겹치는 값 제거 및 합치기
  jeju_insta_df = pd.DataFrame()
  
  f_list = ['1_crawling_jejudoMatJip.xlsx', '1_crawling_jejudoGwanGwang.xlsx', '1_crawling_jejuMatJip.xlsx', '1_crawling_jejuYeoHang.xlsx']
  
  for fname in f_list:
      fpath = './files/' + fname
      temp = pd.read_excel(fpath)
      jeju_insta_df = jeju_insta_df.append(temp)
      
  jeju_insta_df.drop_duplicates(subset = ['content'], inplace = True)
  jeju_insta_df.to_excel('./files/1_crawling_raw.xlsx')
  ```



---

#### 2. 인기 해시태그 값 그래프 그리기

- 각 tag 값 추출

  ```python
  raw_total = pd.read_excel('./files/1_crawling_raw.xlsx')
  
  raw_total['tags'][0][2:-2].split(" ', ' ")
  # 0       ['#제주분식', '#제주맛집', '#제주도맛집', '#제주맛집추천', '#제주도맛...]
  #1       ['#함덕맛집', '#제주도카페투어', '#제주일상', '#함덕', '#jejudo.. 을
  
  # ["#제주분식', '#제주맛집', '#제주도맛집', '#제주맛집추천', '#제주도맛집추천', '#제주도'] 식으로 각각 출력
  
  # 해시태그 하나 하나 값 추출
  tags_total = []
  
  for tags in raw_total['tags']:
      tags_list = tags[2:-2].split("', '")
      for tag in tags_list:
          tags_total.append(tag)
  ```

- 최다 해시태그 값

  ```python
  tag_counts.most_common(50)
  # 50위권 값 계속 체크하면서 불필요한 값 STOPWORDS 리스트에 추가
  STOPWORDS = ['#일상', '#선팔', '#제주도', '#jeju', '#반영구', '#제주자연눈썹',
  '#서귀포눈썹문신', '#제주눈썹문신', '#소통', '#맞팔', '#제주속눈썹', '#제주일상',
               '#눈썹문신', '#여행스타그램','#제주반영구','#제주남자눈썹문신','#서귀포자연눈썹',
   '#서귀포남자눈썹문신','#카멜리아힐','#daily','#제주메이크업', '#셀카']
  
  tag_total_selected = []
  
  for tag in tags_total:
      if not tag in STOPWORDS :
          tag_total_selected.append(tag)
          
  tag_counts_selected = Counter(tag_total_selected)
  ```

- 그래프 그리기

  ```python
  import matplotlib.pyplot as plt
  from matplotlib import rc
  import sys
  import seaborn as sns
  rc('font', family = 'malgun gothic')
  
  tag_count_df = pd.DataFrame(tag_counts_selected.most_common(50))
  tag_count_df.columns = ['tags', 'counts']
  
  ## 해시태그 값 중 blank 지우기
  tag_count_df['tags'].replace('', None, inplace = True)
  tag_count_df.dropna(subset = ['tags'], inplace = True)
  
  plt.figure(figsize = (12, 10))
  sns.barplot(x = 'counts', y = 'tags',
             data = tag_count_df)
  plt.show()
  ```

  