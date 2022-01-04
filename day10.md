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

> 선택 정렬

```
```

