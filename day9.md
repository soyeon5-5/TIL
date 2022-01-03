# 알고리즘 교육

## 1. 선형 자료 구조

- 선형리스트

  ```python
  ## 함수
  def add_data(friend) :
      katok.append(None)
      kLen = len(katok)
      katok[kLen-1] = friend
  
  def insert_data(position, friend) :
      katok.append(None)
      kLen = len(katok)
      for i in range(kLen-1, position, -1) :
          katok[i] = katok[i-1]
          katok[i-1] = None
      katok[position] = friend
  
  def delete_data(position) :
      katok[position] = None
      kLen = len(katok)
      for i in range(position+1, kLen, 1) :
          katok[i-1] = katok[i]
          katok[i] = None
      del(katok[kLen-1])
  
  
  ## 전역
  katok = []
  
  
  ## 메인
  add_data('다현')
  add_data('정연')
  add_data('쯔위')
  add_data('사나')
  add_data('지효')
  print(katok)
  add_data('모모')
  print(katok)
  
  insert_data(3, '미나')
  print(katok)
  
  delete_data(4)
  print(katok)
  ```

- 단순 연결 리스트

  ```python
  ## 함수/클래스 선언부
  
  class Node() :
      def __init__(self):
          self.data = None
          self.link = None
  
  def printNode(start) :
      current = start
      print(current.data, end =' ')
      while current.link != None :
          current = current.link
          print(current.data, end = ' ')
          
  def insertNode(findData, insertData) :  # 첫 노드 앞에 삽입할 때
      global memory, head, current, pre
      if head.data == findData :
          node = Node()
          node.data = insertData
          node.link = head
          head = node
          memory.append(node)
          return
      # 중간 삽입
      current = head
      while current.link != None :
          pre = current
          current = current.link
          if current.data == findData :
              node = Node()
              node.data = insertData
              node.link = current
              pre.link = node
              memory.append(node)
              return
      # 마지막에 추가할 때(=삽입할 이름이 존재하지 않을때)
      node = Node()
      node.data = insertData
      current.link = node
      return
  
  def deleteNode(deleteData) :
      global memory, head, current, pre
      # 첫 노드 삭제
      if deleteData == head.data :
          current = head
          head = head.link
          del(current)
          return
      # 첫 노드 외의 노드 삭제
      current = head
      while current.link != None :
          pre = current
          current = current.link
          if current.data == deleteData :
              pre.link = current.link
              del(current)
              return
  
  def findNode(findData) :
      global memory, head, current, pre
      current = head
      if current.data == findData :
          return current
      while current.link != None :
          current = current.link
          if current.data == findData :
              return current
      return Node()
  
  
  
         
  ## 전역 변수
  memory = []
  head, current, pre = None, None, None
  dataArray = ['다현','정연','쯔위','사나','지효']
  
  
  ## 메인 코드부
  node = Node()     # 첫 노드
  node.data = dataArray[0]
  head = node
  memory.append(node)
  
  for data in dataArray[1:] :
      pre = node
      node = Node()
      node.data = data
      pre.link = node
      memory.append(node)
      
     
  insertNode('다현', '화사')
  printNode(head)
  
  deleteNode('화사')
  printNode(head)
  
  fNode = findNode('지민')
  print(fNode.data)
  ```

- 스택

  기본 개념

  ```python
  ## 전역
  stack = [None, None, None, None, None]
  SIZE = 5
  stack = [None for _ in range(SIZE)]
  top = -1
  
  
  ## 메인
  top += 1
  stack[top] = '커피'
  top += 1
  stack[top] = '녹차'
  top += 1
  stack[top] = '꿀물'
  
  print(stack)
  
  
  data = stack[top]
  stack[top] = None
  top -= 1
  print('팝-->' , data)
  #팝--> 꿀물
  data = stack[top]
  stack[top] = None
  top -= 1
  print('팝-->' , data)
  #팝--> 녹차
  ```

  응용

  ```python
  ## 함수
  def isStackFull() :
      global SIZE, stack, top
      if (top >= SIZE-1) :
          return True
      else :
          return False
  def push(data) :
      global SIZE, stack, top
      if (isStackFull()) :
          print('스택 꽉')
          return
      top += 1
      stack[top] = data
  
  def isStackEmpty() :
      global SIZE, stack, top
      if (top == -1) :
          return True
      else :
          return False
      
  def pop() :
      global SIZE, stack, top
      if (isStackEmpty()) :
          print('스택 텅~')
          return None
      data = stack[top]
      stack[top] = None
      top -= 1
      return data
  
      
  ## 전역
  SIZE = 5
  stack = [ None for _ in range (SIZE)]
  top = -1
  
  ## 메인
  stack = ['커피', '녹차', '꿀물', '콜라', None]
  top = 3
  
  push('맥주')
  print(stack)
  push('포도주')
  print(stack)
  
  stack = ['커피', None, None, None, None]
  top = 0
  print(pop())
  print(pop())
  
  ```

  