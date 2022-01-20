# 22.01.20

## 스타벅스 지도 시각화

#### 1. 파일 병합

```python
# 서울시 시군구별 위도, 경도 파일
seoul_sgg = pd.read_excel('./files/seoul_sgg_list.xlsx')
# 서울시 내 시군구별 스벅 매장수
starbucks_sgg_count = pd.read_excel('./files/seoul_starbuck_sgg_count.xlsx')
# 서울 시군구별 위도/경도+ 스벅 매장수
seoul_sgg = pd.merge(seoul_sgg, starbucks_sgg_count,
                      how = 'left', on = '시군구명')

# 서울 시군구별 인구수
seoul_starbuck_pop = pd.read('./files/sgg_pop.xlsx')
# 위도/경도,스벅매장수 + 인구수
seoul_sgg = pd.merge(seoul_sgg, seoul_starbuck_pop,
                     how = 'left', on = '시군구명')

# 서울 시군구별 종사자수, 사업체수
seoul_starbuck_biz = pd.read('./files/sgg_biz.xlsx')
# 위도/경도,스벅매장수,인구수 + 종사자/사업체 수
seoul_sgg = pd.merge(seoul_sgg, seoul_starbuck_biz,
                     how = 'left', on = '시군구명')
# 저장
seoul_sgg.to_excel('./files/seoul_sgg_stat.xlsx', index = False)
```

#### 2. 지도 시각화

```python
import folium
import json

## 연습
## 스타벅스 매장별 데이터로 지도 시각화
seoul_starbucks = pd.read_excel('./files/seoul_starbucks_list.xlsx')

starbucks_map = folium.Map(location = [37.573050, 126.979189],
                          tiles = 'StamenTerrain',
                          zoom_start = 11)

for idx in starbucks_map.index :
    lat = starbucks_map.loc[idx , '위도']
    lng = seoul_starbucks.loc[idx, '경도']
    store_type = seoul_starbucks.loc[idx, '매장타입']
    names = seoul_starbucks.loc[idx, '매장명']
    address = '<pre>'+seoul_starbucks.loc[idx, '주소']+'</pre>'
    # 한 줄씩 표현 가능해짐
    tel = seoul_starbucks.loc[idx, '전화번호']
    
    fillColor = ''
    if store_type == 'general' :
        fillColor = 'gray'
        
    elif store_type == 'reserve' :
        fillColor = 'blue'
        
    elif store_type == 'generalDT' :
        fillColor = 'red'
    
    folium.CircleMarker(
    location = [lat, lng],
    color = fillColor,
    radius = 4,
    weight = 1, #테두리 두께
    popup = address+tel,
    tooltip = names,
    fill = True,
    fill_color = fillColor,
    fill_opacity = 0.5  # 채워진 색 투명도(1이 불투명)
    ).add_to(starbucks_map)
```

- 지도 구역 표현

  > 국제포맷으로, 원하는 구역 json 검색하여 json 파일 저장

  ```python
  sgg_geojson_file_path = './maps/seoul_sgg.geojson'
  seoul_sgg_geo = json.load(open(sgg_geojson_file_path, encoding = 'utf-8'))
  seoul_sgg_geo['features'][0]['properties']
  '''
  {'SIG_CD': '11320',
   'SIG_KOR_NM': '도봉구',
   'SIG_ENG_NM': 'Dobong-gu',
   'ESRI_PK': 0,
   'SHAPE_AREA': 0.00210990544544,
   'SHAPE_LEN': 0.239901251347}
   '''
  # 이렇게 지역별 구역 범위 저장되어있음
  
  starbucks_bubble = folium.Map(
                      location = [37.573050, 126.979189],
                      tiles = 'CartoDB positron',
                      zoom_start = 11
                      )
  
  def style_function(feature) :
  	return{'opacity' : 0.7,
            'weight' : 1,
            'color' : 'red',
            'fillOpacity' : 0.1,
            'dashArray' : '5, 5'  #점선 오픽셀, 빈칸 오픽셀.. 점선형 표현
            }
  # 구역별 점선으로 구분되어 표현
  ```

- 서울시 구별 스타벅스 매장 수 시각화

  ```python
  # 평균보다 많으면 red, 적거나 같으면 blue
  starbucks_mean = seoul_sgg_stat['스타벅스_매장수'].mean()
  
  for idx in seoul_sgg_stat.index :
      lat = seoul_sgg_stat.loc[idx, '위도']
      lng = seoul_sgg_stat.loc[idx, '경도']
      stat = seoul_sgg_stat.loc[idx, '시군구명']
      count = seoul_sgg_stat.loc[idx, '스타벅스_매장수']
      
      if count > starbucks_mean :
          fillColor = 'red'
          
      else :
          fillColor = 'blue'
          
      folium.CircleMarker(location = [lat, lng],
                         color = 'yellow',
                         radius = count/1,
                          # numpy형식을 못 받음,
                          #int(count) 혹은 count/1 과 같이 int로 바꿔주기
                          weight = 1,
                          popup = count,
                          tooltip = stat,
                          fill_color = fillColor,
                          fill_opacity = 0.7
                         ).add_to(starbucks_bubble)
      
  
  ```

  
