import numpy as np

x = np.array(range(1,101))
print(x)

x2 = np.array(range(100))
print(x2)

x_train = x[0:10]
print(x_train)

x_test = x[10:30]
print(x_test)

# x_train = x[:30]
# print(x_train)

# x_val = x[30:40]
# print(x_val)

# x_test = x[40:]
# print(x_test)