# sin 함수를 그려보자

import numpy as np
import matplotlib.pyplot as plt 

x = np.arange(0, 10, 0.1) # 0부터 10까지 0.1 단위 씩 생성
y = np.sin(x)

plt.plot(x, y)
plt.show()
