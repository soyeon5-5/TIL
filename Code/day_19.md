# 22. 01.18

### 1. instagram 데이터로 장소 COUNT

```python
import pandas as pd
raw_total = pd.read_excel('./files/1_crawling_raw.xlsx')

# 저장된 데이터에서 장소 추출
location_counts = raw_total['place'].value_counts()
location_counts_df = pd.DataFrame(location_counts)
location_counts_df.to_excel('./files/3_3_location_counts.xlsx')

locations = list(location_counts_df.index)
```

- **카카오 검색 API **를 이용한 주소 찾기

  ```python
  import requests
  
  searching = '합정 스타벅스'
  url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query={}'.format(searching)
  
  headers = {
      "Authorization": "KakaoAK 카카오디벨로퍼에서 만든 REST API"
  }
  
  # 합정 스타벅스로 검색되는 장소의 정보
  places = requests.get(url, headers = headers).json()['documents']
  # 장소 이름
  places[1]['place_name']
  # 장소의 위도 경도
  print('경도 =', places[1]['x'])
  print('위도 =',places[1]['y'])
  
  # 함수로 만들기
  def find_places(searching) :
      url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query={}'.format(searching)
      
      headers = {
      "Authorization": "KakaoAK 여기에 REST KEY 쓰기"}
      
      places = requests.get(url, headers = headers).json()['documents']
      
      place = places[0]
      name = place['place_name']
      x = place['x']
      y = place['y']
      
      data = [name, x, y ,searching]
      
      return data
  ```

  

- instagram 장소 별 정보 저장

  ```python
  from tqdm.notebook import tqdm
  #데이터량이 많을 때, 진행 상황 확인
  
  locations_inform = []
  # 연습이라 200개만 함
  for location in tqdm(locations[:200]) :
      try :
          data = find_places(location)
          locations_inform.append(data)
          time.sleep(1)
          
      except :
          pass
      
  
  locations_inform_df = pd.DataFrame(locations_inform)
  locations_inform_df.columns = ['name_official', '경도', '위도', 'name']
  locations_inform_df.to_excel('./files/3_locations.xlsx', index = False)
  ```

- 장소당 count 수와 정보 합치기

  ```python
  locations_inform_df = pd.read_excel('./files/3_locations.xlsx')
  location_counts_df = pd.read_excel('./files/3_location_counts.xlsx', index_col = 0)
  
  location_data = pd.merge(locations_inform_df, location_counts_df,
                          how = 'inner', left_on = 'name_official',
                          right_index = True)
                          
  location_data = location_data.pivot_table(index = ['name_official', '경도', '위도'],
                           values = 'place',
                           aggfunc = 'sum')
                           
  # count 수로 정렬
  location_data.sort_values(by = 'place', ascending=False, inplace = True)
  
  location_data.to_excel('./files/3_location_inform.xlsx', index = False)
  
  ```

####  ## 지도 기초

```python
import folium

Mt_Hanla = [33.3652500, 126.533694]
map_jeju1 = folium.Map(location = Mt_Hanla,
                     zoom_start = 10)

for i in range(len(location_data)) :
    name = location_data['name_official'][i]
    count = location_data['place'][i]
    size = int(count)*2
    long = float(location_data['위도'][i])
    lat = float(location_data['경도'][i])
    folium.CircleMarker((long, lat), radius = size,
                        color = 'red', popoup = name).add_to(map_jeju1)


folium.TileLayer('Stamenterrain').add_to(map_jeju1)
# default는 openstreetmap 임, 변경하고싶을때 입력

map_jeju1
```



### 2. 지도 시각화

```python
location_data = pd.read_excel('./files/3_location_inform.xlsx')

Mt_Hanla = [33.3652500, 126.533694]
map_jeju = folium.Map(location = Mt_Hanla,
                     zoom_start = 10)

for i in range(len(location_data)) :
    name = location_data['name_official'][i]
    count = location_data['place'][i]
    size = int(count)*2
    long = float(location_data['위도'][i])
    lat = float(location_data['경도'][i])
    folium.CircleMarker((long, lat), radius = size,
                        color = 'red', popoup = name).add_to(map_jeju)

# 지도 타입(default는 Stamenterrain)
folium.TileLayer('Stamenterrain').add_to(map_jeju1)

map_jeju
# 원으로 크기 표현되는 지도
```

- 마커 , 클러스터

  ```python
  from folium.plugins import MarkerCluster
  
  Mt_Hanla = [33.3652500, 126.533694]
  map_jeju2 = folium.Map(location = Mt_Hanla,
                       zoom_start = 10)
  
  # 지도 타입 layer 형태로 표현
  tiles = ['stamenwatercolor', 'cartodbpositron', 
           'openstreetmap', 'stamenterrain',
          'Stamentoner', 'cartodbdark_matter',
          ]
  # 참고 블로그 https://blog.naver.com/PostView.nhn?blogId=life4happy&logNo=222157921230
  
  for tile in tiles :
      folium.TileLayer(tile).add_to(map_jeju2)
      
  marker_cluster = MarkerCluster(locations = locations,
                                popups = names,
                                name = 'Jeju',
                                overlay = True,
                                control = True).add_to(map_jeju2)
  
  folium.LayerControl().add_to(map_jeju2)
  
  map_jeju2.save('./files/3_jeju_cluster.html')
  ```

  
