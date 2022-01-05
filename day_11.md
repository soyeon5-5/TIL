# 21.01.05 교육

## 데이터 시각화

### 1. Plot

```python
import csv
import matplotlib.pyplot as plt

f = open('seoul'.csv') # 기온데이터 csv 파일로 다운로드
data = csv.reader(f)
next(data)

result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

        
plt.plot(result, "r")
plt.show()
```

### 2. histogram

```python
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
plt.hist(result, bins =100, color='r')
plt.show()

# 월별 최고 최저 기온

aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08' :
            aug.append(float(row[-1]))
        if month == '01' :
            jan.append(float(row[-1]))

plt.hist(aug, bins = 100, color = 'r', label = 'aug')
plt.hist(jan, bins = 100, color = 'b', label = 'jan')
plt.legend()
plt.show()

# 가로 출력(barh)
plt.barh(range(5), [1,2,3,4,5])
plt.show()
```

### 3. Boxplot

```python
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)

month =[[],[],[],[],[],[],[],[],[],[],[],[]]

for row in data :
    if row[-1] != '':
        month[int(row[0].split('-')[1])-1].append(float(row[-1]))

plt.boxplot(month)
plt.show()

# outliers(이상치) 제거 및 사이즈 조절

day =[]

for i in range(31) :
    day.append([])

for row in data :
    if row[-1] != '' :
        if row[0].split('-')[1] == '05' :
            day[int(row[0].split('-')[2])-1].append(float(row[-1]))

plt.style.use('ggplot')
plt.figure(figsize=(10,5),dpi=300)
plt.boxplot(day, showfliers=False) # outliers 출력 안함
plt.show()
```

---

```python
# 인구 구조 현황 그래프

import csv

f = open('age.csv') # 행정안전부에서 연령별 인구현황 다운로드
data = csv.reader(f)
result = []

name = input('인구 구조가 궁금한 지역의 이름(읍면동 단위)을 입력해주세요 :')

for row in data :
    if name in row[0] :
        for i in row[3:] :
            result.append(int(i))  # 문자열 > 정수 변환
print(result)

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rc('font', family='Malgun Gothic')
plt.title(name+'지역 인구 구조 현황')
plt.plot(result)
plt.show()

# 지역의 성별 인구 분포

import matplotlib.pyplot as plt
import csv
f = open('gender.csv')
data = csv.reader(f)

m = []
f = []
name = input('성별 인구 분포가 궁금한 지역의 이름(읍면동 단위)을 입력해주세요 :')

for row in data :
    if name in row[0] :
        for i in row[3:104] :
            m.append(-int(i))
        for i in row[106:] :
            f.append(int(i))

print(m)
print(f)

plt.style.use('ggplot')
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize = (10,5), dpi=300)
plt.title(name+'의 성별 인구 분포')

plt.barh(range(101), m, label = '남성')
plt.barh(range(101), f, label = '여성')
plt.legend()
plt.show()
```



