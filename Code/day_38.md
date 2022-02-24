# 22.02.17

## ProDS

입문 공부

1. 데이터 세트 내에 총 결측값의 개수는?

   ```python
   data1.isna().sum().sum()
   
   # 참고: 결측치가 포함된 행의수
   data1.isna().any(axis=1).sum()
   ```

2.  TV, Radio, Social Media 등 세 가지 다른 마케팅 채널의 예산과 매출액과의 상관분석을 통하여 각 채널이 매출에 어느 정도 연관이 있는지 알아보고자 한다. 매출액과 가장 강한 상관관계를 가지고 있는 채널의 상관계수를 기술

   ```python
   var_list = ['TV', 'Radio', 'Social_Media', 'Sales' ]
   
   q2 = data1[var_list].corr().abs().drop('Sales')['Sales']
   
   round(q2.max())
   
   # 참고
   q2.nlargest(3) # 상위 3개
   q2.argmax() # 최대값이 있는 위치번호
   q2.idxmax() # 최대값이 있는 인덱스명
   ```

3. 매출액 종속변수, TV, Radio, Social Media 의 예산을 독립변수로 하여 회귀분석 수행하였을 때, 세개의 독립변수의 회귀계수를 큰 것부터 기술

   ```python
   from statsmodels.formula.api import ols
   
   q3 = data1.dropna()
   form1 = 'Sales'+'+'.join(var_list)
   
   ols1 = ols(form1, q3).fit()
   
   ols1.summary() # 전체 정리된 값들
   ols1.params # 회귀계수
   ols1.pvalues # pvalue 값
   
   ols1.params.drop('Intercept').sort_values(ascending=False)
   ```

4. Age, Sex, BP, Cholesterol 및 Na_to_K 값이 Drug 타입에 영향을 미치는지 확인하기 위하여  Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은 ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 초과는 ‘Lv4’로 변환하시오. 

   Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을 수행하시오. 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수 가운데 가장 큰 p-value를 찾아 기술하시오.

   ```python
   q4 = data2.copy()
   
   q4['Age_gr'] = np.where(q2.Age < 20, 10,
                   np.where(q2.Age < 30, 20,
                     np.where(q2.Age < 40, 30,
                       np.where(q2.Age < 50, 40
                         np.where(q2.Age < 60, 50, 60)))))
   
   q4['Na_K_gr']= np.where(q2.Na_to_K <= 10, 'Lv1',
                   np.where(q2.Na_to_K <= 20, 'Lv2',
                   np.where(q2.Na_to_K <= 10, 'Lv3','Lv4')))
   
   var_list = ['Sex', 'BP', 'Cholesterol', 'Age_gr','Na_K_gr']
   
   q4_out2=[]
   for i in var_list:
       temp = pd.crosstab(index=q2[i],
                          columns=q2['Drug'])
   
       q4_out = chi2_contingency(temp)
       chi2 = q4_out[0] # 카이제곱값
       pvalue = q4_out[1]
       q4_out2.append([i, chi2, pvalue])
   
   q4_out2=pd.DataFrame(q4_out2,
                        columns = ['var', 'chi2', 'pvalue'])
   
   # 영향력 있는 변수 추출
   q4_out3 = q4_out2[q4_out2.pvalue < 0.05]
   len(q4_out3)
   
   # 영향력 변수 중에 가장 큰 value
   round(q4_out3.pvalue.max(), 5)
   ```

5. Sex, BP, Cholesterol 등 세 개의 변수를 다음과 같이 변환하고 의사결정나무를 이용한 분석을 수행하시오.
   Sex는 M을 0, F를 1로 변환하여 Sex_cd 변수 생성, BP는 LOW는 0, NORMAL은 1 그리고 HIGH는 2로 변환하여 BP_cd 변수 생성, Cholesterol은 NORMAL은 0, HIGH는 1로 변환하여 Ch_cd 생성
   Age, Na_to_k, Sex_cd, BP_cd, Ch_cd를 Feature로, Drug을 Label로 하여 의사결정나무를 수행하고 Root Node의 split feature와 split value를 기술하시오.

   ```python
   q5 = data2.copy()
   
   q5['Sex_cd'] = np.where(q5.Sex == 'M', 0, 1)
   q5['BP_cd'] = np.where(q5.BP == 'LOW', 0,
                          np.where(q5.BP == 'NORMAL',1 , 2))
   q5['Ch_cd'] = np.where(q5.Cholesterol == 'NORMAL', 0, 1)
   
   # 의사결정나무 실행 -> 모델 생성
   from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
   
   var_list = ['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']
   
   dt = DecisionTreeClassifier().fit(q5[var_list], q5['Drug'])
   
   # Root Node
   plot_tree(dt, max_depth=1, feature_names=var_list,
             class_names=list(q5.Drug.unique()),
             precision=3,
             fontsize=8)
   
   print(export_text(dt, feature_names=var_list, decimals=3))
   
   ```

   
