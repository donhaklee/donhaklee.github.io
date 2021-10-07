# 수치미분

```python
'''
1) 미분하려는 함수 f(x)정의
2) 극한 개념을 구현하기 위해 기울기 x는 작은 값으로 설정
3) 분자/분모구현
'''
# ex1) 함수 f(x) = x^2에서 미분계수 f'(3) 구하기
def numerical_derivative(f,x):
    delta_x = 1e-4
    return (f(x+delta_x) - f(x-delta_x)) / (2*delta_x)

def func1(x):
    return x**2

result = numerical_derivative(func1, 3)
print("f(x) = x^2 : ", result)


# ex2) 함수 f(x) = 3xe^x, f'(2)구하기
import numpy as np
def func2(x):
    return 3*x*(np.exp(x))

def numerical_derivative(f,x):
    delta_x = 1e-4
    return (f(x+delta_x) - f(x-delta_x)) / (2*delta_x)

result = numerical_derivative(func2, 2)
print("f(x) = 3xe^x : ", result)

```

    ex1) f(x) = x^2 :  6.000000000012662
    ex2) f(x) = 3xe^x :  66.50150507518049

---

## 수치미분 : 다변수 함수 (최종버전)

```python
import numpy as np
def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad


# ex3) 2변수 함수 예제 f(x,y) = 2x + 3xy + y^3, f'(1.0, 2.0)
def func3(input_obj):
    x = input_obj[0]
    y = input_obj[1]
    return (2*x + 3*x*y + np.power(y,3))

input = np.array([1.0, 2,0])
numerical_derivative(func3, input)

# ex4) 4변수 함수 예제 f(w,x,y,z) = wx + xyz + 3w + zy^2
# f'(1.0, 2.0, 3.0, 4.0)
def func4(input_obj):
    w = input_obj[0,0]
    x = input_obj[0,1]
    y = input_obj[1,0]
    z = input_obj[1,1]
    return (w*x + x*y*z + 3*w + z*np.power(y,2))

input = np.array([[1.0, 2.0], [3.0, 4.0]])
numerical_derivative(func4, input)

```

```sh
ex3) array([ 8.        , 15.00000001,  0.        ])

ex4) array([[ 5., 13.],
            [32., 15.]])
```

