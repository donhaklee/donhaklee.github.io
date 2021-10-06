
---
layout: post
title:  "Matplotlib"
---
# Matplotlib


```python
# matplotlib
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
x_data = np.random.rand(100)
y_data = np.random.rand(100)

plt.title('scatter plot')
plt.grid()
plt.scatter(x_data,y_data, color='b',marker='o')
plt.show()
```


    
![png](output_11_0.png)
    



```python
#line plot
x_data = [x for x in range(-5,5)]
y_data = [y*y for y in range(-5,5)]
plt.title('line plot')
plt.grid()
plt.plot(x_data, y_data, color='b')
plt.show()
```


    
![png](output_12_0.png)
    
