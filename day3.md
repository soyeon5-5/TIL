# 21.12.23 교육

## 파이썬 교육

```python
.sum(axis=0)   # 행별(각 열의 서로 다른 행끼리),x=0
.sum(axis=1)   # 열별(각 행의 서로 다른 열끼리),x=1

df1.iloc[:,0].sum()
df1.iloc[:,0].mean()

df1.iloc[0,0] = np.nan
df1.iloc[:,0].mean() 

df1.iloc[:,0].isnull()  # 조건(boolean)

df1.iloc[:,0][df1.iloc[:,0].isnull()] = df1.iloc[:,0].mean()

#데이터 프레임 전체에서  NaN 값 확인
df1[df1.notnull()] 
df1[df1.isnull()]

df1.iloc[:,0].var()   #분산
df1.iloc[:,0].std()   #표준편차
df1.iloc[:,0].min()   #최소값
df1.iloc[:,0].median()  #중위수(중앙값)

df1
(df1.iloc[:,0] >= 10).sum()   #조건에 만족하는 개수 확인

```

---

## LINUX

1. **Oracle VM VirtulBox** 설치
2. **centos Linux** 설치
3. **Virtual box** 설정
4. **linux** 생성
5. **게스트 확장 cd** 실행 및 설치
