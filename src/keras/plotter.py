import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

x = np.linspace(-5 , 5,100) # 100 linearly spaced numbers
y = np.maximum(x, 0) # computing the values of sin(x)/x
# y = np.tanh(x)
# y = sigmoid(x)

# compose plot
plt.plot(x,y, linewidth=3) # sin(x)/x
# plt.plot(x,y,'co') # same function with cyan dots
# plt.plot(x,2*y,x,3*y) # 2*sin(x)/x and 3*sin(x)/x
plt.savefig("relu.pdf")
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.grid()

# plt.show() # show the plot