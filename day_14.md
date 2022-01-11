# 22.01.11

## 1. Naive Bayes & Confusion Matrix

> - precision(정밀도) : 예측 기준 TP, 진짜 양성인 것을 맞추는것
>
> - recall(재현율,민감도) : 실제(Actual)기준 TP, 진짜 양성인것을 맞추는 것 
>
> - f1-score : 정밀도와 재현율의 조화 평균

```python
import numpy as np
import pandas as pd
from nltk.corpus import stopwords #stopword : 불용어(the, a, 등)
import string
import nltk
import csv
df = pd.read_csv('spam.csv')
```

#### 	1 ) 베이지안 확률 정의

```python
def process_text(text):
    # 구두점 제거
    # text 에서 무의미한 단어(접미사, 조사 등) 삭제 --> stopwords(불용어) 제거
    nopunc = [char for char in text if char not in string.punctuation] # list comprehension
    nopunc = ''.join(nopunc)
    
    # 소문자로 전부 변환(대/소문자 구분)
    cleaned_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    return cleaned_words
```

	#### 	2) data 변환

```python
# text 행렬 변환(token 수)
from sklearn.feature_extraction.text import CountVectorizer

messages_bow = CountVectorizer(analyzer = process_text).fit_transform(df['text'])

#data를 train, test로 분류
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['label_num'], test_size = 0.2, random_state = 0)
```

#### 	3) naive_bayes 다항식 모델 사용

```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train) # train 데이터로 fit 해야함

# 예측값, 실제값 출력
print(classifier.predict(X_train)) # 예측값
print(y_train.values) # 실제값
```

#### 	4) 학습 데이터셋 모델 정확도

```python

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

pred = classifier.predict(X_train) # 예측값 출력

# metrics pkg(패키지)에 있는 정밀도(precision), 재현율(sensitivity), f-1 score 구하기 
print(classification_report(y_train, pred)) # 실제값, 예측값 비교

confusion_matrix(y_train,  pred) #실제값, 예측값

accuracy_score(y_train, pred)
```

#### 	5) 테스트 셋에서 모델 정확도

```python
# 테스트 데이터 셋 모델의 정확도 평가
classifier.predict(X_test)

# 실제 관측값 출력
print(y_test.values)

# 테스트 셋에서 모델 평가
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

pred = classifier.predict(X_test) # X_test : test data(새로운 데이터 간주)
print(confusion_matrix(y_test, pred)) # 실제 라벨, 예측 라벨
print(accuracy_score(y_test, pred)) # 예측률 
```



---

## 2. ROC Curve

```
```

