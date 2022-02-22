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

   
