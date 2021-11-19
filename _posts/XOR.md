# XOR 구현과정
- 논리게이트는 손실함수로 Cross-entropy를 이용해서 Classification으로 결과를 예측할 수 있음
### 1. External function
- def sigmoid ( 0 또는 1을 출력하기 위한 함수 )
- def numerical_derivative(f,x)

### 2. LogicGate class
```python
class LogicGate:
  def __init__(self, gate_name, xdata, tdata) # __xdata, __tdata, __W, __b초기화
  def __loss_func(self) # 손실함수 cross-entropy
  def error_val(self) # 손실함수 값 계산
  def train(self) # 수치미분을 이용하여 손실함수 최소값을 찾는 메소드
  def predict(self, xdata) # 미래값 예측 메소드
```

### 3. usage
```python
xdata = np.array([ [0,0], [0,1], [1,0], [1,1] ])  # 입력데이터 생성
tdata = np.array([0,0,0,1]) # 정답데이터 생성

AND_obj = LogicGate("AND_GATE", xdata, tdata) # LogicGate 객체생성
AND_obj.train() # 손실함수 최소값을 갖도록 학습

AND_obj.predict(...) # 임의 데이터에 대해 결과 예측
```

---
## XOR 구현
```python
(1) sigmoid 함수와 수치미분 함수 구현
import numpy as np
# sigmoid 함수
def sigmoid(x):
    return 1 / (1+np.exp(-x))
# 수치미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)
        x[idx] = tmp_val - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        x[idx] = tmp_val 
        it.iternext()   
    return grad
  
(2) LogicGate class

class LogicGate:
    def __init__(self, gate_name, xdata, tdata):  # xdata, tdata => numpy.array(...)
        self.name = gate_name
        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4,2) # private으로 표현
        self.__tdata = tdata.reshape(4,1)
        
        # 가중치 W, 바이어스 b 초기화
        self.__W = np.random.rand(2,1)  # weight, 2 X 1 matrix
        self.__b = np.random.rand(1)
                        
        # 학습률 learning rate 초기화
        self.__learning_rate = 1e-2
        
    # 손실함수 (classification 내용과 동일)
    def __loss_func(self):
        delta = 1e-7    # log 무한대 발산 방지
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        # cross-entropy 
        return  -np.sum( self.__tdata*np.log(y + delta) + (1-self.__tdata)*np.log((1 - y)+delta ) )      
    
    # 손실 값 계산 (classification 내용과 동일)
    def error_val(self):
        delta = 1e-7    # log 무한대 발산 방지
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        # cross-entropy 
        return  -np.sum( self.__tdata*np.log(y + delta) + (1-self.__tdata)*np.log((1 - y)+delta ) )

    # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
    def train(self):
        f = lambda x : self.__loss_func()
        print("Initial error value = ", self.error_val())
        for step in  range(8001):
            self.__W -= self.__learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            if (step % 400 == 0):
                print("step = ", step, "error value = ", self.error_val())
                 
    # 미래 값 예측 함수
    def predict(self, input_data):
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)
        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False
        return y, result
```

```python
(3) usage (AND)
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 0, 0, 1])

AND_obj = LogicGate("AND_GATE", xdata, tdata)

AND_obj.train()


# AND Gate prediction
print(AND_obj.name, "\n")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data) 
    print(input_data, " = ", logical_val, "\n") 



(3-1) usage (OR)
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 1])

AND_obj = LogicGate("OR_GATE", xdata, tdata)

AND_obj.train()


# AND Gate prediction
print(OR_obj.name, "\n")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data) 
    print(input_data, " = ", logical_val, "\n") 


```
<br>
XOR은 손실함수 값이 2.7근처에서 더이상 감소하지 않는다. <br>
=> classification으로 구현 불가능 <br><br>

### => NAND, OR, AND조합으로 구현해야한다.

```python
# XOR 을 NAND + OR => AND 조합으로 계산함
input_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

s1 = []    # NAND 출력
s2 = []    # OR 출력

new_input_data = []  # AND 입력
final_output = []    # AND 출력

for index in range(len(input_data)):
    
    s1 = NAND_obj.predict(input_data[index])  # NAND 출력
    s2 = OR_obj.predict(input_data[index])    # OR 출력
    
    new_input_data.append(s1[-1])    # AND 입력
    new_input_data.append(s2[-1])    # AND 입력
    
    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))
    
    final_output.append(logical_val)    # AND 출력, 즉 XOR 출력    
    new_input_data = []    # AND 입력 초기화


for index in range(len(input_data)):    
    print(input_data[index], " = ", final_output[index], end='')
    print("\n")

```


![Random](https://github.com/donhaklee/donhaklee.github.io/blob/9620c6585a0418c726078721635ed158da9e6904/images/XOR.PNG)<br><br>

