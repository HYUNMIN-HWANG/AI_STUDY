# Overfit
# Dropout

"""
[과적합 (overfit)]
> train data로 그래프를 그릴 때 너무 딱 드러맞게 그리면 이후 test, validation data를 할 때 오히려 오차가 많이 생긴다.
> 훈련시킬 때 버릴 놈은 버리는 게 오히려 더 낫다.
> 데이터를 버리고 train시킬 경우 train의 성능이 조금 떨어질 수 있어도 train과 test 간의 차이가 줄어든다.

[과적합을 방지하기 위해서]
1. 훈련 데이터를 늘린다.
2. 피쳐 (속성, 열)을 줄인다. (필요한 열만 남긴다)
3. regularization
4. Dropout (딥러닝에서 사용됨)

5?. 앙상블 (논란 : 통상 2~5% 향상이 있다하더라)

"""
