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
- Multi-Variable
### 2. Logistic Regression (Classification)
- Cross-Entropy
- Multi-Variable
---
# 1. Linear Regression
## 1) Loss function
#### 손실함수(loss function)
![Random](lossfunction.png)
- 모든 데이터에 대한 평균 오차 값
- training data의 정답(t)와 입력(x)에 대한 계산 값 y의 차이를 모두 더해 수식으로 나타낸 것
- E(W,b) = (t-[Wx+b])^2을 사용
- E(W,b)가 작다는 것은 평균 오차가 작다는 의미고 미지의 데이터 x가 주어질 경우 확률적으로 미래의 결과값도 오차가 작다고 추측
- 최종목적 : E(W,b)가 최소값을 갖도록 (W,b)를 구하는 것이 Linear Regression
---
## 2) Gradient Decent Algorithm

---
## 3) Multi-Variable Linear Regression


---
# 2. Logistic Regression - Classification
## 1) Cross-entropy

---
## 2) Multi-Variable
