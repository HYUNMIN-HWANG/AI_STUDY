import numpy as np

def sigmoid (x) :
    return 1 / (1+np.exp(-x))


def relu(x) :
    return np.maximum(0, x)

alpa = 0.5
def elu (x) :
    return (x>=0)*x + (x<0)*alpa(np.exp(x)-1)

scale = 1.0507
a = 1.67633
def selu(x) :
    return scale * (np.maximum(0,x) + np.minimum(0, a*(np.exp(x)-1)))

