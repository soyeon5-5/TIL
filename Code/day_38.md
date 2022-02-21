# 22.02.17

## ProDS

입문 공부

1. 매출액 종속변수, TV, Radio, Social Media 의 예산을 독립변수로 하여 회귀분석 수행하였을 때, 세개의 독립변수의 회귀계수를 큰 것부터 기술

   ```python
   from statsmodels.formula.api import ols
   
   q1 = data1.dropna()
   form1 = 'Sales'+'+'.join(var_list)
   
   ols1 = ols(form1, q1).fit()
   
   ols1.summary() # 전체 정리된 값들
   ols1.params # 회귀계수
   ols1.pvalues # pvalue 값
   
   ols1.params.drop('Intercept').sort_values(ascending=False)
   ```

2. Age, Sex, BP, Cholesterol 및 Na_to_K 값이 Drug 타입에 영향을 미치는지 확인하기 위하여  Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은 ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 초과는 ‘Lv4’로 변환하시오. 

   Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을 수행하시오. 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수 가운데 가장 큰 p-value를 찾아 기술하시오.

   ```python
   q2 = data2.copy()
   
   q2['Age_gr'] = np.where(q2.Age < 20, 10,
                   np.where(q2.Age < 30, 20,
                     np.where(q2.Age < 40, 30,
                       np.where(q2.Age < 50, 40
                         np.where(q2.Age < 60, 50, 60)))))
   
   q2['Na_K_gr']= np.where(q2.Na_to_K <= 10, 'Lv1',
                   np.where(q2.Na_to_K <= 20, 'Lv2',
                   np.where(q2.Na_to_K <= 10, 'Lv3','Lv4')))
   
   var_list = ['Sex', 'BP', 'Cholesterol', 'Age_gr','Na_K_gr']
   
   q2_out2=[]
   for i in var_list:
       temp = pd.crosstab(index=q2[i],
                          columns=q2['Drug'])
   
       q2_out = chi2_contingency(temp)
       chi2 = q2_out[0] # 카이제곱값
       pvalue = q2_out[1]
       q2_out2.append([i, chi2, pvalue])
   
   q2_out2=pd.DataFrame(q2_out2,
                        columns = ['var', 'chi2', 'pvalue'])
   
   # 영향력 있는 변수 추출
   q2_out3 = q2_out2[q2_out2.pvalue < 0.05]
   len(q2_out3)
   
   # 영향력 변수 중에 가장 큰 value
   round(q2_out3.pvalue.max(), 5)
   ```

3. Sex, BP, Cholesterol 등 세 개의 변수를 다음과 같이 변환하고 의사결정나무를 이용한 분석을 수행하시오.
   Sex는 M을 0, F를 1로 변환하여 Sex_cd 변수 생성, BP는 LOW는 0, NORMAL은 1 그리고 HIGH는 2로 변환하여 BP_cd 변수 생성, Cholesterol은 NORMAL은 0, HIGH는 1로 변환하여 Ch_cd 생성
   Age, Na_to_k, Sex_cd, BP_cd, Ch_cd를 Feature로, Drug을 Label로 하여 의사결정나무를 수행하고 Root Node의 split feature와 split value를 기술하시오.

   ```python
   q3 = data2.copy()
   
   q3['Sex_cd'] = np.where(q3.Sex == 'M', 0, 1)
   q3['BP_cd'] = np.where(q3.BP == 'LOW', 0,
                          np.where(q3.BP == 'NORMAL',1 , 2))
   q3['Ch_cd'] = np.where(q3.Cholesterol == 'NORMAL', 0, 1)
   
   # 의사결정나무 실행 -> 모델 생성
   from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
   
   var_list = ['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']
   
   dt = DecisionTreeClassifier().fit(q3[var_list], q3['Drug'])
   
   # Root Node
   plot_tree(dt, max_depth=1, feature_names=var_list,
             class_names=list(q3.Drug.unique()),
             precision=3,
             fontsize=8)
   
   print(export_text(dt, feature_names=var_list, decimals=3))
   
   ```

   
