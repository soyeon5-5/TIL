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
  ```

  