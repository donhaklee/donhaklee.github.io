# 지도학습 (supervised)
## 회귀 : Regression
### 1. Linear Regression
- Loss function
- Gradient Decent
### 2. Logistic Regression (Classification)
- Cross-Entropy
---
# 1. Linear Regression
- Training Data를 이용하여 데이터의 특성과 상관관계 등을 파악하고 그 결과를 바탕으로 
Training Data에 없는 미지의 데이터가 주어졌을 경우에, 그 결과를 연속적인 숫자 값으로 예측하는 것
- ex) 공부시간과 시험성적간의 관계, 집 평수와 집 가격간의 관계 등
- input => learning => ask => predict
- y = Wx + b에서 가중치(기울기) W와 y절편 bias를 구하는 개념
- 오차 = t-(Wx+b)로 계산되며 오차가 크다면 우리가 임의로 설정한 직선의 가중치와 바이어스 값이 잘못된 것이고 오차가 작다면 직선의 가중치와 바이어스 값이 잘 된 것
## 1) Loss function
#### 손실함수(loss function)
![Random](https://github.com/donhaklee/donhaklee.github.io/blob/14d4843fda29eb857edc42542bd8567be2dcec6d/images/lossfunction.PNG)
- 모든 데이터에 대한 평균 오차 값
- training data의 정답(t)와 입력(x)에 대한 계산 값 y의 차이를 모두 더해 수식으로 나타낸 것
- E(W,b) = (t-[Wx+b])^2을 사용
- E(W,b)가 작다는 것은 평균 오차가 작다는 의미고 미지의 데이터 x가 주어질 경우 확률적으로 미래의 결과값도 오차가 작다고 추측
- 최종목적 : E(W,b)가 최소값을 갖도록 (W,b)를 구하는 것이 Linear Regression
---
## 2) Gradient Decent Algorithm
### 최소값 찾는 법
- 임의의 가중치 W선택
- 그 W에서의 직선의 기울기를 나타내는 미분 값을 구함 (해당 W에서의 미분)
- 그 미분 값이 작아지는 방향으로 W를 감소(증가)시켜나가면
- 최종적으로 기울기가 더 이상 작아지지 않는 곳을 찾을 수 있는데 그 곳이 손실함수 E(W) 최소값임을 알 수 있음
- 이처럼 W에서의 직선의 기울기인 미분 값을 이용하여 그 값이 작아지는 방향으로 진행하여 손실함수 최소값을 찾는 방법을 경사하강법이라고 함<br><br>
![Random](https://github.com/donhaklee/donhaklee.github.io/blob/9620c6585a0418c726078721635ed158da9e6904/images/gradientdecent.PNG)<br><br>
- 편미분 값이 양수일 때는 현재의 W에서 편미분 값만큼 빼줘서 감소시켜야하고
- 편미분 값이 음수일 때는 현재의 W에서 편미분 값만큼 더해줘서 증가시켜야한다.<br><br>

![Random](https://github.com/donhaklee/donhaklee.github.io/blob/c7893ce350a4331548476842bf776e59860bad13/images/LinearRegressionProcess.PNG)

---
## 3) Single & Multi-Variable Linear Regression
### 코딩 단계
(1) 슬라이싱 또는 list comprehension을 이용하여 입력 x와 정답 t를 numpy데이터형으로 분리<br>
(2) W = numpy.random.rand(...), b = numpy.random.rand(...)
```python
(3) regression 손실함수 정의
# X, W, t, y 모두 numpy행렬
def loss_func(...) :
  y = numpy.dot(X,W) + b # 행렬곱
  return ( numpy.sum((t-y)**2)) / (len(x))

(4) 수치미분, 학습률 알파 : learning_rate = 1e-3, or 1e-4 or 1e-5

(5) 가중치 W, 바이어스 b 를 업데이트하며 최소값 구하기
f = lambda x : loss_func(...)
for step in range(6000) : # 6000은 임의값
  W -= learning_rate * numerical_derivative(f, W)
  b -= learning_rate * numerical_derivative(f, b)
```
변환 행렬 식 : X * W + b = Y

### Single variable 예제

```python
# (1) 학습데이터 준비
import numpy as np
x_data = np.array([1,2,3,4,5]).reshape(5,1)
t_data = np.array([2,3,4,5,6]).reshape(5,1)

# (2) 임의의 직선 y = Wx + b정의 (임의의 값으로 가중치 W, 바이어스 b 초기화)
W = np.random.rand(1,1)
b = np.random.rand(1)
print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

# (3) 손실함수 정의
def loss_func(x, t) :
  y = np.dot(x,W) + b
  return (np.sum((t-y) ** 2)) / (len(x))
  
# (4) 수치미분 및 utility함수 정의
def numerical_derivative(f,x) :
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished :
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

def error_val(x,t) : # 손실함수 값 계산 함수, 입력변수 x t (numpy type)
  y = np.dot(x,W) + b
  return (np.sum((t-y) ** 2)) / (len(x))

def predict(x) :
  y = np.dot(x, W) + b
  return y

# (5) 가중치 W, 바이어스 b를 업데이트하며 최소값 구하기
learning_rate = 1e-2
f = lambda x : loss.func(x_data, t_data)
print("Initial error value = ", error_val(x_data, t_data), "initial W = ", W, "\n", ", b = ", b)
for step in range(8001):
  W -= learning_rate * numerical_derivative(f, W)
  b -= learning_rate * numerical_derivative(f, b)
  if( step % 400 == 0):
    print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, ", b = ",b)

```
<br>

### multi variable 예제
x1W1 + x2W2 + x3W3 + b
X * W + b = Y
```python
# (1) 학습데이터 준비
import numpy as np
loaded_data = np.loadtxt('./data-01-test-score.csv', delimiter=',', dtype = np.float32)
x_data = loaded_data[ :, 0:-1 ] # 모든행에 대하여 1열부터 3열까지 슬라이싱을 통해 입력데이터로 가져옴
t_data = loaded_data[ :, [-1] ] # 정답 데이터는 모든행에 대하여 4열의 데이터를 정답데이터로 정함

# (2) 임의의 직선 y = W1x1 + W2x2 + W3x3 + b정의
W = np.random.rand(3,1) # 3x1행렬
b = np.random.rand(1)
print("W= ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

# (3) 손실함수 E(W,b) 정의
def loss_func(x, t):
  y = np.dot(x, W) + b
  return ( np.sum((t-y) ** 2)) / (len(x) )

# (4) 수치미분 및 utility함수 정의 (single과 동일)
def numerical_derivative(f,x) :
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished :
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
    
def error_val(x,t) : # 손실함수 값 계산 함수, 입력변수 x t (numpy type)
  y = np.dot(x,W) + b
  return (np.sum((t-y) ** 2)) / (len(x))

def predict(x) :
  y = np.dot(x, W) + b
  return y

# (5) 가중치 W, 바이어스 b를 업데이트하며 최소값 구하기 (single과 동일)
learning_rate = 1e-2
f = lambda x : loss.func(x_data, t_data)
print("Initial error value = ", error_val(x_data, t_data), "initial W = ", W, "\n", ", b = ", b)
for step in range(8001):
  W -= learning_rate * numerical_derivative(f, W)
  b -= learning_rate * numerical_derivative(f, b)
  if( step % 400 == 0):
    print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, ", b = ",b)

```


---
# 2. Logistic Regression - Classification
- Training Data 특성과 관계 등을 파악 한 후에 미지의 입력 데이터에 대해서 결과가 어떤 종류의 값으로 분류될 수 있는지를 예측하는 것
- ex) 스팸문자 분류 [Spam(1) or Ham(0)], 암 판별 [악성종양(1) or 종양(0)]
- input => learning => ask => predict
- Training Data 특성과 분포를 나타내는 최적의 직선을 찾고 그 직선을 기준으로 데이터를 위(1) 또는 아래(0) 등으로 분류해주는 알고리즘
- 이러한 Logistic Regression은 Classification 알고리즘 중에서도 정확도가 높은 알고리즘으로 알려져 있어서 Deep Learning에서 기본 Component로 사용됨
- (x, t) => Regression(Wx+b) => classification(sigmoid) => true(1), false(0)
- sigmoid 계산 값이 0.5보다 크면 결과로 1이 나올 확률이 높다는 것이기 때문에 출력 값 y는 1을 정의하고
- sigmoid 계산값이 0.5미만이면 결과로 0이 나올 확률이 높다는 것이므로 출력 값 y는 0정의하여 classification 시스템을 구현할 수 있음



## 1) Cross-entropy
손실함수 (Cross-entropy) : 분류시스템 최종 출력 값 y는 sigmoid함수에 의해 논리적으로 1 또는 0값을 가지기 때문에 연속 값을 갖는 선형회귀 때와는 다른 손실함수가 필요함
- 가중치 W와 bias는 수치미분으로 구할 수 있음
![Random](https://github.com/donhaklee/donhaklee.github.io/blob/1c514ab2a51602a19c03e8112d6a715b9a0c2c3a/images/ClassificationLossFunction.PNG)
- classification 최종 출력 값 y는 sigmoid함수에 의해 0~1 사이의 값을 갖는 확률적인 분류 모델이므로, 다음과 같이 확률변수 C를 이용해 출력 값을 나타낼 수 
![Random](https://github.com/donhaklee/donhaklee.github.io/blob/7b5dd2446d487b46d03e73d6a9a8e07b684ffca2/images/plus.PNG)

![Random](https://github.com/donhaklee/donhaklee.github.io/blob/1c514ab2a51602a19c03e8112d6a715b9a0c2c3a/images/ClassificationProcess.PNG)
---
## 2) Multi-Variable
