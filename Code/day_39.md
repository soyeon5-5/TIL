# 22.02.18

## ProDS

입문 공부

1. Chance_of_Admit 확률이 0.5를 초과하면 합격으로, 이하이면 불합격으로 구분하고 로지스틱 회귀분석을 수행하시오.
   원데이터만 사용하고, 원데이터 가운데 Serial_No와 Label은 모형에서 제외, 각 설정값은 다음과 같이 지정하고, 언급되지 않은 사항은 기본 설정값을 사용하시오
   Seed : 123
   로지스틱 회귀분석 수행 결과에서 로지스틱 회귀계수의 절대값이 가장 큰 변수와 그 값을 기술하시오. 

   ```python
   import numpy as np
   import pandas as pd
   
   data1 = pd.read_csv(' .csv', na_values=['?',' ','NA'])
   
   q1 = data1.copy()
   
   q1['Ch_cd'] = np.where(q1.Chance_of_Admit > 0.5, 1, 0)
   
   from sklearn.linear_model import LogisticRegression
   
   x_list = data1.columns.drop(['Serial_No', 'Chance_of_Admit'])
   logit=LogisticRegression(fit_intercept=False,
                           random_state=123,
                           solver='liblinear')
   logit.fit(q1[x_list], q1[Ch_cd])
   q1_out = pd.Series(logit.coef_.reshape(-1))
   q1_out.index = x_list
   q1_out.abs().nlargest(1)
   ```

2. 독립변수로 RandD_Spend, Administration, Marketing_Spend를 사용하여 Profit을 주별로 예측하는 회귀 모형을 만들고, 이 회귀모형을 사용하여 학습오차를 산출하시오.
   주별로 계산된 학습오차 중 MAPE 기준으로 가장 낮은 오차를 보이는 주는 어느 주이고 그 값은 무엇인가?  (MAPE = Σ ( | y - y ̂ | / y ) * 100/n )

   ```python
   state_list= data2.State.uniqe()
   var_list = ['RandD_Spend', 'Adiministration', 'Marketing_Spend']
   
   from sklearn.linear_model import LinearRegression
   
   q2_out =[]
   for i in state_list:
       temp=data2[data2.State == i]
       lm=LinearRegression().fit(temp[var_list], temp['Profit'])
       pred=lm.predict(temp[var_list])
       mape=(abs(temp['Profit'] - pred)/temp['Profit']).sum()*100/len(temp)
       
   q2_out = pd.DataFrame(q2_out, columns=['state', 'mape'])
   
   q2_out.loc[q2_out.mape.idxmin(), :]
   ```

3. 경력과 최근 이직시 공백기간의 상관관계를 보고자 한다. 남여별 피어슨 상관계수를 각각 산출하고 더 높은 상관계수를 기술하시오.

   ```python
   data3.goupby('gender')[['experience','last_new_job']].corr()
   ```

4. 기존 데이터 분석 관련 직무 경험과 이직 의사가 서로 관련이 있는지 알아보고자 한다. 이를 위해 독립성 검정을 실시하고 해당 검정의 p-value를 기술하시오.
   검정은 STEM 전공자를 대상으로 한다.
   검정은 충분히 발달된 도시(도시 개발 지수가 제 85 백분위수 초과)에 거주하는 사람을 대상으로 한다. 이직 의사 여부(target)은 문자열로 변경 후 사용한다.

   ```python
   # 1. 데이터 타입 변경
   q3 = data3.copy()
   q3['target'] = q3['target'].astype(str)
   
   # 2. 조건으로 데이터 필터링
   q3['major_discipline'].value_count()
   
   base=q3['city_development_index'].quantile(0.85)
   
   q3_1 = q3[(q3['major_discipline']=='STEM')&
            (q3['city_development_index']>base)]
   
   # 3. 범주형 데이터 독립성 검정 : 카이스퀘어
   from scipy.stats import chi2_contingency
   
   q3_tab = pd.crosstab(index=q3_1.relevent_experience,
                       columns=q3_1.target)
   q3_out = chi2_contingency(q3_tab)[1]
   ```

   
