# 22.03.10

## Tensorflow

#### Advanced use of RNN

- 날짜 시계열 데이터로 24시간 후 온도 예측

  1. 데이터 준비

     ```python
     # jena_climate_2009_2016.csv.zip 다운받아오기
     !unzip 'jena_climate_2009_2016.csv.zip'
     ```

     ```python
     import os
     
     data_dir = './jena_climate'
     fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
     
     f = open(fname)
     data = f.read()
     f.close()
     
     lines = data.split('\n')
     header = lines[0].split(',')
     lines = lines[1:]
     
     print(header) 
     print(len(lines))
     ```

     ```python
     #parsing
     import numpy as np
     
     float_data = np.zeros((len(lines), len(header) - 1))
     
     for i, line in enumerate(lines): 
         values = [float(x) for x in line.split(',')[1:]]
         float_data[i, :] = values
         
     # 시계열 데이터 확인
     from matplotlib import pyplot as plt
     
     temp = float_data[:, 1] 
     plt.plot(range(len(temp)), temp)
     
     # 10일간 데이터
     plt.plot(range(1440), temp[:1440])
     ```

  2. 데이터 전처리

     ```python
     mean = float_data[:200000].mean(axis=0)
     float_data -= mean 
     std = float_data[:200000].std(axis=0)
     float_data /= std
     ```

     



