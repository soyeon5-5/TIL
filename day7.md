# 21.12.29 교육

## 1. 날짜 표현

- 현재 날짜

  ```python
  import numpy as np
  import pandas as pd
  from pandas import Series, DataFrame
  
  from datetime import datetime
  datetime.now()
  
  d1 = datetime.now()
  d1.year     # 연
  d1.month    # 월
  d1.day      # 일
  
  ```

- 날짜 포맷 변경

  날짜 포맷 변경 후 return data type은 문자

  ```python
  d1 = datetime.now()
  datetime.strftime(d1, '%A')  #완전체
  datetime.strftime(d1, '%a')  #축약형
  # 요일 리턴  Wednesday
  datetime.strftime(d1, '%m-%d,%Y')
  # 12-29,2021
  ```

- 날짜 연산

  ```python
  d1      # 현재 날짜
  # 1) offset
  from pandas.tseries.offsets import Day, Hour, Second
  d1+Day(100)
  #Timestamp('2022-04-08 11:43:03.597602')
  
  # 2) timedelta (날짜와의 차이)
  from datetime import timedelta
  d1+timedelta(100)
  #datetime.datetime(2022, 4, 8, 11, 43, 3, 597602)  d1형식 유지
  
  # 3) DataOffset 
  d1 + pd.DateOffset(month = 4)
  
  # 4) 날짜 - 날짜
  d2='2022/01/01'
  d3 = d1 - datetime.strptime(d2, '%Y/%m/%d')
  d3.days
  d3.seconds
  ```



---

## 2. 데이터 시각화

- 선 그래프

  ```python
  import numpy as np
  import pandas as pd
  from pandas import Series, DataFrame
  
  import matplotlib.pyplot as plt
  
  
  s1.plot(xticks=[0,1,2,3],  # 눈금 좌표
          ylim=[0,100],      # y축 범위
          xlabel='x name',   # x축 라벨
          ylabel='y name',   # y 축 라벨
          rot=90,            # rot= rotation 회전 90도
          title='name',      # title
          marker='^',        # 마커
          linestyle='--',    # 선 스타일
          color='red')       # 선 색상
          
  
  # global option 변경
  plt.rc('font', family='Malgun Gothic')
  
  # 데이터 프레임의 선 그래프 출력
  df1 = DataFrame({'apple':[10,20,30,40],'banana':[49,39,30,12],'mango':[10,32,43,40]})
  
  df1.index = ['a','b','c','d']
  df1.index.name = '지점'
  df1.columns.name = '과일명'
  plt.legend(fontsize=9, loc='best', title='과일 이름')
  ```

- Bar Plot

  ```python
  kimchi= pd.read_csv('./code/kimchi_test.csv',encoding='cp949')
  kimchi = kimchi.pivot_table(index='판매월', columns='제품', values='수량', aggfunc='sum')
  
  kimchi.plot(kind='bar')
  plt.title('김치별 판매수량 비교')
  plt.ylim(0,300000)
  plt.ylabel('판매수량')
  plt.legend(fontsize=9, loc='best', title='김치별')
  plt.xticks(rotation=0)
  ```

- Pie Plot(원형 차트)

  ```python
  plt.pie(ratio,                  # 각 파이 숫자
          labels=labels,          # 각 파이 이름
          autopct='%.1f%%',       # 값의 표현 형태(소수점 첫째자리)
          startangle=260,         # 시작위치
          radius = 0.8,           # 파이 크기
          counterclock=False,     # 시계방향 진행 여부
          explode = explode,      # 중심에서 벗어나는 정도 설정(각각 서로 다른 숫자 전달 가능)
          colors=colors,          # 컬러맵 전달 가능
          shadow=False,           # 그림자 설정
          wedgeprops=wedgeprops)  # 부채꼴 모양 설정
          
  # make data
  x = [1, 2, 3, 4]
  colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))
  
  # plot
  fig, ax = plt.subplots()
  ax.pie(x, colors=colors, radius=3, center=(4, 4),
         wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
  
  ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
         ylim=(0, 8), yticks=np.arange(1, 8))
  
  plt.show()
  ```

- hist :  히스토그램(밀도표현)

  ```python
  s1 = Series(np.random.rand(1000))  
  s1.hist()# 정해진 숫자에서 무작위 추출(균등->uniform distribution)
  
  s1 = Series(np.random.randn(1000))  # 정규분포(normal distribution)에서 무작위 추출
  s1.hist(bins=4) # 막대의 수(계급 구간 전달)
  
  plt.hist(s1,
           bins=5,
           density=True)   # True 로 설정시, 막대 아래 총 면적이 1이 되는 밀도 함수 출력
                           # 즉, y축 값이 확률로 변경되어 출력됨
                           
  plt.hist(s1, density=False) # 확률 값으로 출력
  s1.plot(kind='kde')  # 커널 밀도 함수(kernel density estimation) 출력
                     # 연속형 히스토그램
  ```

- scatter(산점도)

  ```python
  # iris data loading
  from sklearn.datasets import load_iris
  
  iris = load_iris()
  iris.keys()
  #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
  
  iris['DESCR']
  iris['data']
  iris_x=iris['data']
  
  iris['feature_names']
  x_names= iris['feature_names']
  
  # plt.scatter(iris_x[:,0], iris_x[:,1])
  # iris_x[:,0] 는 x축 좌표(첫번째 설명 변수)
  # iris_x[:,1] y축 좌표(두번째 설명 변수)
  # c = iris_x[:,1] color 설정 안할 경우 단일색으로 출력
  #      color 설정 시 서로 다른 숫자 전달시, 서로 다른색 표현(채도)
  
  # 봄, 여름, 가을, 겨울 색상으로 4가지 표현
  
  plt.subplot(2,2,1)  # 2*2 그래프 중 1번째
  plt.scatter(iris_x[:,0], iris_x[:,1], c=iris_x[:,1])
  plt.spring() # spring색상으로 변경
  plt.xlabel(x_names[0])
  plt.ylabel(x_names[1])
  plt.colorbar()
  
  plt.subplot(2,2,2) # 2*2 그래프 중 2번째
  plt.scatter(iris_x[:,1], iris_x[:,2], c=iris_x[:,2]) 
  plt.summer()
  plt.xlabel(x_names[1])
  plt.ylabel(x_names[2])
  plt.colorbar()
  
  plt.subplot(2,2,3) # 2*2 그래프 중 3번째
  plt.scatter(iris_x[:,2], iris_x[:,3], c=iris_x[:,3]) 
  plt.autumn()
  plt.xlabel(x_names[2])
  plt.ylabel(x_names[3])
  plt.colorbar()
  
  plt.subplot(2,2,4) # 2*2 그래프 중 4번째
  plt.scatter(iris_x[:,3], iris_x[:,0], c=iris_x[:,0]) 
  plt.winter()
  plt.xlabel(x_names[0])
  plt.ylabel(x_names[0])
  plt.colorbar()
  ```

- box plot

  ```
  plt.boxplot(iris_x)
  plt.xticks(ticks=[1,2,3,4], labels=x_names)
  ```

  

