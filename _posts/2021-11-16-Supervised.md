# 지도학습 (supervised)
## 회귀 : Regression
- Training Data를 이용하여 데이터의 특성과 상관관계 등을 파악하고 그 결과를 바탕으로 
Training Data에 없는 미지의 데이터가 주어졌을 경우에, 그 결과를 연속적인 숫자 값으로 예측하는 것<br>
- ex) 공부시간과 시험성적간의 관계, 집 평수와 집 가격간의 관계 등 <br>
- input => learning => ask => predict
- y = Wx + b에서 가중치(기울기) W와 y절편 bias를 구하는 개념
- 오차 = t-(Wx+b)로 계산되며 오차가 크다면 우리가 임의로 설정한 직선의 가중치와 바이어스 값이 잘못된 것이고 오차가 작다면 직선의 가중치와 바이어스 값이 잘 된 것
### 1. Linear Regression
- Loss function
- Gradient Decent
### 2. Logistic Regression (Classification)
- Cross-Entropy
---
# 1. Linear Regression
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
## 3) Multi-Variable Linear Regression
### 코딩 단계
- 
```python


```

---
# 2. Logistic Regression - Classification
## 1) Cross-entropy

---
## 2) Multi-Variable
