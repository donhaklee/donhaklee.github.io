# 딥러닝 (Deep Learning)

## 1. 신경망 게념
![Random](https://github.com/donhaklee/donhaklee.github.io/blob/f050ad7e16288ddc349a4522c374934feafcfb3d/images/neural.PNG)<br><br>

단일로 Regression과 classification을 하면 XOR을 구현할 수 없다 <br>
그렇기에 다중으로 구현해야하는데 이런 구현형태는 신경망과 닮아있다.<br>

- 신경 세포 뉴런은 이전 뉴런으로부터 입력신호를 받아 또 다른 신호를 발생시킨다. <br>
- 그러나 입력에 비례해서 출력을 내는 형태(y = WX)가 아니라 입력 값들의 모든 합이 어느 임계점에 도달해야만 출력 신호를 발생시킨다 <br>
- 이처럼 입력신호를 받아 특정 값의 임계점을 넘어서는 경우에 출력을 생성해주는 함수를 활성화 함수(activation function)이라고 하는데 지금까지 사용해왔던 classification 시스템의 sigmoid 함수가 대표적인 활성화 함수이다.
- 즉, sigmoid에서의 임계점은 0.5로서 입력값 합이 0.5보다 크면 1을 출력으로 내보내고 0.5보다 값이 작으면 출력을 내보내지 않는다고 볼 수 있다.

![Random](https://github.com/donhaklee/donhaklee.github.io/blob/32dee95435156eb65d5ec6165e916c0de9de6232/images/activation.PNG)<br><br>

## <신경세포인 뉴런 동작원리를 머신러닝에 적용시키기 위한 과정>
1. 입력신호와 W를 곱하고 적당한 b를 더한 후 (Linear Regression)
2. 그 값을 활성화 함수(sigmoid) 입력으로 전달 (classification)해서 sigmoid 함수 임계점 0.5를 넘으면 1을 그렇지 않으면 0을 다음 뉴런으로 전달해주는 multi-variable Logistic Regression 시스템 구축

=> 인공 신경망 : Logistic Regression을 여러개 연결 시켜 출력하는 것

---

## 2. 딥러닝 개념

![Random](https://github.com/donhaklee/donhaklee.github.io/blob/991947e7e784f66cce6239eaa56f0112df1de82e/images/deep.PNG)<br><br>
- 노드 : 1개의 Logistic Regression
- 노드가 서로 연결되어 있는 신경망 구조를 바탕으로 입력층(Input Layer), 1개 이상의 은닉층(Hidden Layer), 출력층(Output Layer)에서의 오차를 기반으로 각 노드(뉴런)의 W를 학습시키는 머신러닝 분야
- 딥러닝 구조에서의 1개 이상의 은닉층을 이용하여 학습시키면 정확도가 높은 결과를 얻을 수 있다고 알려져있음
- 은닉층을 깊게 할수록 정확도가 높아진다고 해서 Deep Learning이라는 용어가 사용됨
- 예시 : W21 : 특정 계층 노드 1에서 다음 은닉층의 노드 2로 전달되는 신호를 강화 또는 약화시키는 가중치
