import math

def sigmoid(x, deff=False):
  if deff:
    return sigmoid(x)*(1-sigmoid(x))
  else:
    return 1 / (1 + math.exp(-x))

print(sigmoid(0.04, False))
print(sigmoid(0.17, False))
print(sigmoid(0.604835, False))
print(sigmoid(0.63357, False))


print(sigmoid(0.04000040984, False))
print(sigmoid(0.1760001386, False))
print(sigmoid(0.249407, False))
print(sigmoid(0.6056402, False))
print(sigmoid(0.6342834751, False))

print(sigmoid(0.0488792, False))
print(sigmoid(0.1788805, False))
print(sigmoid(0.614904228, False))
print(sigmoid(0.6358434751, False))
