# 22. 01. 21

## 다나와 에서 상품 비교

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
```



#### 1. 원하는 제품 crawling(무선청소기)

```python
driver = webdriver.Chrome(service = Service('../chromedriver/chromedriver.exe'))

url = 'http://search.danawa.com/dsearch.php?k1=무선청소기'
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
# 제품 정보 확인
'''
prod_items = soup.select('div.main_prodlist > ul.product_list > li.prod_item ')
len(prod_items)
prod_items
prod_items[0].select('a.click_log_product_standard_title_')[0].text
title = prod_items[0].select('p.prod_name > a')[0].text
spec_list = prod_items[0].select('div.spec_list')[0].text.strip()
price = prod_items[0].select('li.rank_one > p.price_sect > a > strong')[0].\
text.strip().replace(',', "")
print(title, spec_list, price)
'''
```

- 제품 당 데이터 저장 함수

  ```python
  def get_prod_items(prod_items):
      prod_data = []
  
      for prod_item in prod_items :
          try :
              title = prod_item.select('p.prod_name > a')[0].text
          except :
              title : ""
  
          try :
              spec_list = prod_item.select('div.spec_list')[0].text.strip()
          except :
              spec_list = ""
  
          try :
              price = prod_item.select('li.rank_one > p.price_sect > a > strong')[0].\
      text.strip().replace(',', "")
          except :
              price = 0
  
          mylist = [title, spec_list, price]
  
          prod_data.append(mylist)
  
      
      return(prod_data)
  
  ## 여러페이지 확인 후 반복 되는 부분 keyword(다나와는 상품이름)
  def get_search_page_url(keyword, page) :
      url =  'http://search.danawa.com/dsearch.php?query={0}&originalQuery={1}&volumeType=allvs&page={2}&limit=40&sort=saveDESC&list=list&boost=true&addDelivery=N&recommendedSort=Y&defaultUICategoryCode=12215657&defaultPhysicsCategoryCode=1824%7C15247%7C221461%7C0&defaultVmTab=3747&defaultVaTab=522985&tab=goods'.format(keyword, keyword, page)
      return url
  ```

  - 여러 페이지 값 가져오기

    ```python
    import time
    from tqdm import tqdm_notebook
    
    keyword = '무선청소기'
    total_page = 10
    prod_data_total = []
    
    for page in tqdm_notebook(range(1, total_page+1)):
        url = get_search_page_url(keyword, page)
        driver.get(url)
        
        time.sleep(5)
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        prod_items = soup.select('div.main_prodlist > ul.product_list > li.prod_item ')
        prod_item_list = get_prod_items(prod_items)
        
        prod_data_total = prod_data_total + prod_item_list
    ```

- 데이터 프레임

  ```python
  data = pd.DataFrame(prod_data_total)
  data.columns = ['상품명', '스펙목록', '가격']
  
  data.to_excel('./files/1_danawa_crawling_result(청소기).xlsx',
               index = False)
  ```

- columns 별 정리

  - 상품명(회사, 상품이름)

    ```python
    company_list = []
    product_list = []
    
    for title in data['상품명'] :
        try :
            title_info = title.split(' ', 1)
            company_name = title_info[0]
            product_name = title_info[1]
    
        
        except :
            company_name = None
            product_name = None
            
        company_list.append(company_name)
        product_list.append(product_name)
    ```

  - 스펙 목록(카테고리, 사용시간, 흡입력)

    ```python
    category_list = []
    use_time_list = []
    suction_list = []
    
    for spec_data in data['스펙목록'] :
        spec_list = spec_data.split(' / ')
                          
        category = spec_list[0].strip()
        category_list.append(category)
        
        for spec in spec_list :
            if '사용시간' in spec :
                use_time_spec = spec
            elif '흡입력' in spec :
                suction_spec = spec
        
        use_time_value = use_time_spec.split(': ')[1].strip()
        suction_value = suction_spec.split(': ')[1].strip()
        use_time_list.append(use_time_value)
        suction_list.append(suction_value)
    ```

    - 사용시간 통일

      ```python
      def convert_time_minute(time) :
          
          try :
              if '시간' in time :
                  hour = time.split('시간')[0]
                  if '분' in time :
                      minute = time.split('시간')[-1].split('분')[0]
                  else :
                      minute = 0
              else :        
                  hour = 0
                  minute = time.split('분')[0]
              
              return int(hour) * 60 + int(minute)
              
          except :
              return None
              
      
      new_use_time_list = []
      for time in use_time_list :
          value = convert_time_minute(time)
          new_use_time_list.append(value) 
      ```

    - 흡입력 단위 통일

      ```python
      def get_suction(value) :
          try :
              value = value.upper()
              
              if 'AW' in value or 'W' in value :
                  result = value.replace('A', '').replace('W', '').replace(',', '')
                  result = int(result)
                  
              elif 'PA' in value :
                  result = value.replace('PA', '').replace(',', '')
                  result = int(result)/100
                  
              else :
                  result = None
              return result
              
          except :
              return None
              
      
      new_suction_list = []
      
      for power in suction_list :
          value = get_suction(power)
          new_suction_list.append(value)
      ```

  - None 값 drop

    ```python
    data.replace('', np.nan, inplace = True)
    data.dropna(inplace = True)
    ```

- 데이터프레임 수정

  ```
  pd_data = pd.DataFrame()
  pd_data['카테고리'] = category_list
  pd_data['회사명'] = company_list
  pd_data['제품'] = product_list
  pd_data['가격'] = data['가격'].astype(np.int64)
  pd_data['사용시간'] = new_use_time_list
  pd_data['흡입력'] = new_suction_list
  ```

- 조건

  ```python
  condition = pd_data['카테고리'] == '핸디/스틱청소기'
  
  pd_data_final = pd_data[condition]
  len(pd_data_final)
  '''
  pd_data_final['가격'] = pd_data_final['가격'].replace('', np.nan)
  pd_data_final.dropna(inplace = True)
  # 워닝 참고자료
  #https://velog.io/@cjw9105/Python-SettingWithCopyWarning-%EC%9B%90%EC%9D%B8
  '''
  pd_data_final.dropna(inplace = True)
  
  # 저장 및 불러오기
  pd_data_final.to_excel('./files/2_danawa_data_final.xlsx', index = False)
  danawa_data = pd.read_excel('./files/2_danawa_data_final.xlsx')
  
  # 평균값
  price_mean_value = danawa_data['가격'].mean()
  suction_mean_value = danawa_data['흡입력'].mean()
  use_time_mean_value = danawa_data['사용시간'].mean()
  
  # 조건
  condition_data = danawa_data [
      (danawa_data['가격'] <= price_mean_value) & 
      (danawa_data['흡입력'] >= suction_mean_value) & 
      (danawa_data['사용시간'] >= use_time_mean_value)]
  ```

- **그래프**

  ```python
  from matplotlib import font_manager, rc
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  rc('font', family = 'Malgun Gothic')
  
  
  plt.figure(figsize = (20, 10))
  sns.scatterplot(x = '흡입력', y = '사용시간',
                 size = '가격', hue = danawa_data['회사명'],
                 data = danawa_data, legend = False,
                 sizes = (10, 1000))
  plt.hlines(use_time_mean_value, 0, 400, color = 'red',
            linestyle = 'dashed', linewidth = 1)
  plt.vlines(suction_mean_value, 0, 120, color = 'red',
           linestyle = 'dashed', linewidth = 1)
  
  plt.show()
  ```

- 사용시간, 흡입력 기준 top20 **그래프**

  ```python
  top_list = danawa_data.sort_values(["사용시간","흡입력"], ascending = False)
  
  chart_data_selected = top_list[:20]
  
  plt.figure(figsize=(20, 10))
  plt.title("무선 핸디/스틱청소기 TOP 20")
  sns.scatterplot(x = '흡입력', 
                    y = '사용시간', 
                    size = '가격', 
                    hue = chart_data_selected['회사명'], 
                    data = chart_data_selected, sizes = (100, 2000),
                    legend = False)
  
  for index, row in chart_data_selected.iterrows():
      x = row['흡입력']
      y = row['사용시간']
      s = row['제품'].split(' ')[0]
      plt.text(x, y, s, size=20)
      
  plt.show()
  ```

  