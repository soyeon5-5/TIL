# python pandas groupby

- 그룹연산

```python
import pandas as pd
from pandas import Series, DataFrame

#pd.read_csv("절대경로",인코딩)
kimchi = pd.read_csv("./kimchi_test.csv", encoding = 'cp949')
#kimchi.groupby(by=None, # 그룹핑 할 컬럼(기준)
#               axis=0,  # 그룹핑 연산 방향
#               level: None) # 멀티 인덱스일 경우 특정 레벨의 값을 그룹핑 컬럼으로 사용

kimchi.groupby(by=['제품','판매처'])['수량'].sum()
# 제품 기준, 수량과 판매금액 총 합 구하기

kimchi.groupby(by=['제품'])[['수량','판매금액']].sum()
# 이중리스트
```

- agg

  ```python
  kimchi.groupby(by=['제품','판매처'])[['수량','판매금액']].agg(['sum','mean'])
  # 제품별, 판매처별(김치별) 수량 판매금액 총합,평균
  
  kimchi.groupby(by=['제품','판매처'])[['수량','판매금액']].agg({'수량':'sum', '판매금액':'mean'})
  #제품별, 판매처별(김치별) 수량은 총합만, 판매금액은 평균
  ```

- multi index

  ```python
  kimchi_2 = kimchi.groupby(by=['제품','판매처'])['수량'].sum()
  kimchi_2.groupby(level=0).sum() # 제품별 총합
  kimchi_2.groupby(level='제품').sum()
  kimchi_2.groupby(level=1).sum() # 판매처별 총합
  ```

  