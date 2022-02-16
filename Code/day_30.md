# 22.02.07

## Scikit-Learn

```python
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
```

- scale

  ```python
  x = (np.arange(10, dtype=np.float) - 3).reshape(-1,1)
  
  df = pd.DataFrame(np.hstack([x, scale(x), robust_scale(x), minmax_scale(x), maxabs_scale(x)]),
                    columns=["x", "scale(x)", "robust_scale(x)", "minmax_scale(x)", "maxabs_scale(x)"])
  ```

  - iris data 이용

    ```python
    from sklearn.datasets import load_iris
    iris = load_iris()
    data1 = iris.data
    data2 = scale(iris.data)
    
    print("old mean:", np.mean(data1, axis = 0))
    print("old std:", np.std(data1, axis =0))
    print("new mean:", np.mean(data2, axis =0))
    print("new std:", np.std(data2, axis =0))
    ```

    