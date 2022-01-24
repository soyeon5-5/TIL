# 알고리즘 교육
## 2. 비선형 자료 구조

### 1) 트리

> 이진 탐색 트리

```python
## 함수
class TreeNode() :
    def __init__(self):
        self.left = None
        self.data = None
        self.right = None

## 전역
memory = []
root = None
nameAry = ['블랙핑크', '레드벨벳','마마무', '에이핑크', '걸스데이', '트와이스']

## 메인
node = TreeNode()
node.data = nameAry[0]
root = node
memory.append(node)

for name in nameAry[1:] :  # ['레드벨벳','마마무', '에이핑크', '걸스데이', '트와이스']
    node = TreeNode()
    node.data = name
    current = root
    while True :
        if (current.data > name) :
            if (current.left == None) :
                current.left = node
                break
            current = current.left
        else :
            if (current.right == None) :
                current.right = node
                break
            current = current.right
    memory.append(node)

print('이진 탐색 트리 구성 완료!')

findName = '마마무'

current = root
while True :
    if current.data == findName :
        print(findName, '찾았음. 야호')
        break
    elif current.data > findName :
        if current.left == None :
            print(findName, '이 트리에 없어요')
            break
        current = current.left
    else :
        if current.right == None :
            print(findName, '이 트리에 없어요')
            break
        current = current.right
# 이진 탐색 트리 구성 완료!
# 마마무 찾았음. 야호
```

### 2) 그래프

> 개념, 연관 행렬

```python
# 4x4 무방향 그래프 
## 함수
class Graph() :
    def __init__(self, size) :
        self.SIZE = size
        self.graph = [[0 for _ in range(size)] for _ in range(size)]

## 전역
G = None

## 메인
G = Graph(4)

G.graph[0][1] = 1
G.graph[0][2] = 1
G.graph[0][3] = 1
G.graph[1][0] = 1
G.graph[1][2] = 1
G.graph[1][3] = 1
G.graph[2][0] = 1
G.graph[2][1] = 1
G.graph[2][3] = 1
G.graph[3][0] = 1
G.graph[3][1] = 1
G.graph[3][2] = 1

print('## 무방향 그래프 ##')
for row in range(4) :
    for col in range(4) :
        print(G.graph[row][col], end = ' ')
    print()
```



---

## 3. 알고리즘

### 1) 정렬

- 선택 정렬 1

  ```python
  import random
  ## 선택정렬1 (가장 쉬운 정렬이지만, 실제로 써도 됨)
  ## 함수
  def findMinIndex(ary) :
      minIdx = 0
      for i in range(1, len(ary)) :
          if (ary[minIdx] > ary[i]) :
              minIdx = i
      return minIdx
  
  ## 전역
  before = [random.randint(1, 99) for _ in range(20)]
  after = []
  
  ## 메인
  print('정렬 전 -->', before)
  for i in range(len(before)) :
      minPos = findMinIndex(before)
      after.append(before[minPos])
      del(before[minPos])
  
  print('정렬 후 -->', after)
  ```

- 선택 정렬 2

  ```python
  import random
  ## 함수
  def selectionSort(ary) :
      n = len(ary)
      for i in range(0, n-1) :
          minIdx = i
          for k in range(i+1, n):
              if (ary[minIdx] > ary[k]) :
                  minIdx = k
          ary[i], ary[minIdx] = ary[minIdx], ary[i]  
      return ary
  
  ## 전역
  dataAry = [random.randint(1, 99) for _ in range(20)]
  
  ## 메인
  print('정렬전 -->', dataAry)
  dataAry = selectionSort(dataAry)
  print('정렬후 -->', dataAry)
  ```

### 2) 검색

- 순차 검색

  ```python
  import random
  ## 함수
  def seqSearch(ary, fData) :
      pos = -1
      size = len(ary)
      for i in range(size) :
          if ary[i] == fData :
              pos = i
              break    
      return pos
  
  ## 전역
  dataAry = [random.randint(1,99) for _ in range(20)]
  findData = dataAry[random.randint(0,len(dataAry)-1)]
  
  ## 메인
  print('배열-->', dataAry)
  position = seqSearch(dataAry, findData)
  if position == -1 :
      print(findData, '없습니다.')
  else :
      print(findData, ':', position, '위치에 있음')
  ```

- 이진 검색

  ```python
  import random
  ## 함수
  def binarySearch(ary, fData) :
      pos = -1
      start = 0
      end = len(ary) - 1
      while (start <= end) :
          mid = (start + end) // 2
          if fData == ary[mid] :
              return mid
          elif fData > ary[mid] :
              start = mid + 1
          else :
              end = mid - 1  
      return pos
  
  ## 전역
  dataAry = [random.randint(1,99) for _ in range(20)]
  findData = dataAry[random.randint(0,len(dataAry)-1)]
  dataAry.sort()
  
  ## 메인
  print('배열-->', dataAry)
  position = binarySearch(dataAry, findData)
  if position == -1 :
      print(findData, '없습니다.')
  else :
      print(findData, ':', position, '위치에 있음')
  ```

### 3) 재귀

```python
## 함수
def openBox() :
    global count
    print('상자 열기 ~~')
    count -= 1
    if count == 0:
        print('## 반지 넣기 ##')
        return
    openBox()
    print('!! 상자 닫기')
    return

## 메인
count =5
openBox()


# 예시. 우주선 발사 카운트
def countDown(n) :
    if n == 0 :
        print('발사')
    else :
        print(n)
        countDown(n-1)
        
countDown(5)
```

