# 21.12.23 교육

```python
.sum(axis=0)   # 행별(각 열의 서로 다른 행끼리),x=0
.sum(axis=1)   # 열별(각 행의 서로 다른 열끼리),x=1

df1.iloc[:,0].sum()
df1.iloc[:,0].mean()

df1.iloc[0,0] = np.nan
df1.iloc[:,0].mean() 


```

