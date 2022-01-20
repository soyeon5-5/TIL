# 22.01.19

## 동네 스타벅스 매장

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By

import pandas as pd
import numpy as np
import time
```



#### 1. 스타벅스 매장 crawling 및 저장

```python
ser = Service('../chromedriver/chromedriver.exe')
driver = webdriver.Chrome(service = ser)

url = 'https://www.starbucks.co.kr/store/store_map.do?disp=locale'
driver.get(url)

# 서울 클릭
seoul_btn = '#container > div > form > fieldset > div > section > article.find_store_cont > article > article:nth-child(4) > div.loca_step1 > div.loca_step1_cont > ul > li:nth-child(1) > a'
driver.find_element(By.CSS_SELECTOR, seoul_btn).click()

# 전체 클릭
all_btn = '#mCSB_2_container > ul > li:nth-child(1) > a'
driver.find_element(By.CSS_SELECTOR, all_btn).click()
time.sleep(3)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# 매장 정보
starbucks_soup_list = soup.select('.quickSearchResultBoxSidoGugun > .quickResultLstCon')

# 매장 정보 list 저장
starbucks_list = []

for item in starbucks_soup_list :
    name = item.select('strong')[0].text.strip()
    lat = item['data-lat']
    lng = item['data-long']
    store_type = item('i')[0]['class'][0][4:]]
    address = str(item.select('p')[0]).split('<br/>')[0].split('>')[1]
    tel = str(item.select('p')[0]).split('<br/>')[1].split('<')[0]

    mylist = [name, lat, lng, store_type, address, tel]
    
    starbucks_list.append(mylist)
    
# 데이터프레임 후 파일 저장
columns = ['매장명', '위도', '경도', '매장타입', '주소', '전화번호']
seoul_starbucks_df = pd.DataFrame(starbucks_list, columns = columns)
seoul_starbucks_df.to_excel('./files/seoul_starbucks_list.xlsx', index = False)
```



#### ## 서울열린데이터광장 OPEN API 활용

```python
import requests
SEOUL_API_AUTH_KEY = '********'
service = 'GangseoListLoanCompany'
url = 'http://openapi.seoul.go.kr:8088/{}/json/{}/1/5'.format(SEOUL_API_AUTH_KEY, service)


# 리스트 형식이 아니므로 일단 블랭크(empty)로 만듦

def seoul_open_api_data(url, service) :
    data_list = None
    
    try :
        result_dict = requests.get(url).json()
        result_data = result_dict[service]
        code = result_data['RESULT']['CODE']
        
        if code == 'INFO-000' :
            data_list = result_data['row']
    except :
        pass
    
    return(data_list)
```

- 서울시의 시군구별 인구수 값 저장(현재는 안됨)

  ```python
  # report.txt 값에 서울시 행정구역 시군구별 인구수 데이터 저장됨
  sgg_pop_df = pd.read_csv('./files/report.txt',
                          header = 2,
                          sep = '\t')
                          
  # sgg_pop_df.columns
  columns = {
      '기간': 'GIGAN',
      '자치구': 'JACHIGU',
      '계': 'GYE_1',
      '계.1': 'GYE_2',
      '계.2': 'GYE_3',
      '남자': 'NAMJA_1',
      '남자.1': 'NAMJA_2',
      '남자.2': 'NAMJA_3',
      '여자': 'YEOJA_1',
      '여자.1': 'YEOJA_2',
      '여자.2': 'YEOJA_3',
      '세대': 'SEDAE',
      '세대당인구': 'SEDAEDANGINGU',
      '65세이상고령자': 'N_65SEISANGGORYEONGJA'
  }
  # rename은 컬럼 수가 달라도 가능
  sgg_pop_df.rename(columns = columns, inplace = True)
  
  
  # 시군구명, 주민등록 총 인구 만 저장
  condition = sgg_pop_df['JACHIGU'] != '합계'
  sgg_pop_df_select = sgg_pop_df[condition]
  
  columns = ['JACHIGU', 'GYE_1']
  sgg_pop_df_final = sgg_pop_df_select[columns]
  sgg_pop_df_final.columns = ['시군구명', '주민등록인구']
  
  sgg_pop_df_final.to_excel('./files/sgg_pop.xlsx', index = False)
  ```

- 서울시 행정구역별 사업체 수 저장

  ```python
  sgg_biz_df = pd.read_csv('./files/report2.txt',
                          header = 2,
                          sep = '\t')
  
  columns = {
      '기간': 'GIGAN',
      '자치구': 'JACHIGU',
      '동': 'DONG',
      '사업체수': 'SAEOPCHESU_1',
      '여성대표자': 'YEOSEONGDAEPYOJA',
      '계': 'GYE',
      '남': 'NAM',
      '여': 'YEO'
  }
  sgg_biz_df.rename(columns = columns, inplace = True)
  sgg_biz_df_select = sgg_biz_df[sgg_biz_df['DONG'] == '소계']
  
  
  # 시군구명, 종사자수, 사업체수 저장
  columns = ['JACHIGU', 'GYE', 'SAEOPCHESU_1']
  sgg_biz_df_final = sgg_biz_df_select[columns]
  sgg_biz_df_final.columns = ['시군구명', '종사자수','사업체수']
  
  sgg_biz_df_final = sgg_biz_df_final.reset_index(drop = True)
  
  sgg_biz_df_final.to_excel('./files/sgg_biz.xlsx', index = False)
  ```



#### 2. 지역별 스타벅스 매장 수

```python
# 스타벅스 매장 정보 불러오기
seoul_starbucks = pd.read_excel('./files/seoul_starbucks_list.xlsx')

# 시군구명 컬럼 추가
sgg_names = []

for address in seoul_starbucks['주소'] :
    sgg = address.split()[1]
    sgg_names.append(sgg)
    
seoul_starbucks['시군구명'] = sgg_names

seoul_starbucks.head()

seoul_starbucks.to_excel('./files/seoul_starbucks_list.xlsx', index = False)

seoul_starbuck = pd.read_excel('./files/seoul_starbucks_list.xlsx')

# 시군구명당 매장수
seoul_starbucks_sgg_count = seoul_starbuck.pivot_table(values = '매장수',
                                                      index = '시군구명',
                                                      aggfunc = 'count')
seoul_starbuck_sgg_count = seoul_starbuck_sgg_count.rename(columns = {'매장명' : '스타벅스_매장수'})

seoul_starbucks_sgg_count.to_excel('./files/seoul_sgg_count')

#지도 시각화는 21일에 이어서
```

