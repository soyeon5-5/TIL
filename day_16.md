# 22.01.13

## 음악차트 순위 Crawling

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd

browser = Service('../chrome_driver/chromedirver.exe')
driver = webdriver.Chrome(service = browser)
```

### 1. 멜론 차트

```python
url = 'https://www.melon.com/chart/index.htm' # 멜론차트 페이지
driver.get(url)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# 100위 차트 크롤링해서 엑셀 저장
# 페이지마다 다르므로 늘 확인
songs = soup.select('tbody > tr')
# song = songs[0] 으로 생각하고 for문 전에 한 번씩 확인

rank = 1
song_data = []
for song in songs :
    title = song.select('div.ellipsis.rank01 > span > a')[0].text
    singer = song.select('div.ellipsis.rank02 > a')[0].text
    mylist = ['melon', rank, title, singer]
    song_data.append(mylist)
    rank += 1
    
# 엑셀 저장
columns = ['서비스', '순위', '타이틀', '가수']
pd_data = pd.DataFrame(song_data, columns = columns)
pd_data.to_excel('./files/melon.xlsx', index = False)
```

### 2. 유튜브

```python
url = 'https://youtube-rank.com/board/bbs/board.php?bo_table=youtube'
driver.get(url)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

channels = soup.select('tr.aos-init')
channel_rank = []
for channel in channels :
    category = channel.select('p.category')[0].text.strip()
    title = channel.select('h1 > a')[0].text.strip()
    sub = channel.select('.subscriber_cnt')[0].text
    view = channel.select('.view_cnt')[0].text
    video = channel.select('.video_cnt')[0].text
    channel_rank.append([category, title, sub, view, video])
    
columns = ['카테고리', '채널명', '구독자 수', 'view 수', '비디오 수']
pd_data = pd.DataFrame(channel_rank, columns = columns)
pd_data.to_excel('./files/youtube.xlsx', index = False)
```

### 3. 엑셀파일 합치기

```python
excel_name = ['./files/youtube.xlsx',
             './files/melon.xlsx']

append_data = pd.DataFrame()

for name in excel_name :
    pd_data = pd.read_excel(name)
    append_data = append_data.append(pd_data)
    
append_data.to_excel('./files/append_data.xlsx', index = False)
```