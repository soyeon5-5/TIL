# 22.01.12

## 웹 크롤링 기초

> selenium 인스톨, bs4 인스톨, 크롬드라이버 설치

```python
from selenium import webdriver
from selenium.wedbriver.chrome.service import Service
from bs4 import BeautifulSoup

ser = Service('../chrome_dirver/chromedirver/exe')
driver = webdriver.Chrome(service = ser)

# ex1
url = 'https://www.naver.com'
driver.get(url)
# 크롬창 생성

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')


# ex2
html = '''
<html>
    <head>
    </head>
    <body>
        <h1> 우리동네시장</h1>
            <div class = 'sale'>
                <p id='fruits1' class='fruits'>
                    <span class = 'name'> 바나나 </span>
                    <span class = 'price'> 3000원 </span>
                    <span class = 'inventory'> 500개 </span>
                    <span class = 'store'> 가나다상회 </span>
                    <a href = 'http://bit.ly/forPlaywithData' > 홈페이지 </a>
                </p>
            </div>
            <div class = 'prepare'>
                <p id='fruits2' class='fruits'>
                    <span class ='name'> 파인애플 </span>
                </p>
            </div>
    </body>
</html>
'''
# 특정 data 가져오기
soup = BeautifulSoup(html, 'html.parser')
soup.select('span')
soup.select('#fruits1') # id는 #
soup.select('.price') # calss는 .(dot)
soup.select('span.name')
- 하위값
soup.select('#fruits1 > span.name')
soup.select('div.sale > #fruits1 > span.name')
soup.select('div.sale span.name')
# 세 data 모두 동일

# text 반환
- 여러 data가 있는 경우
tags_a = soup.select('a')
tags = tags_a[0]
tags.text   # '메뉴' 출력
tags['href']   # 'http://bit.ly/forPlaywithData' 출력
```



