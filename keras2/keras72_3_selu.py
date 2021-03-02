# selu 함수
# https://pytorch.org/docs/stable/generated/torch.nn.SELU.html

import numpy as np
import matplotlib.pyplot as plt

scale = 1.0507
alpha = 1.6733

def selu(x) :
    return scale * (np.maximum(0,x)+np.minimum(0,alpha*(np.exp(x)-1)))

x = np.arange(-5, 5, 0.1)
y = selu(x)

print(x)
print(y)

# 시각화
plt.plot(x, y)
plt.grid()
plt.show()
