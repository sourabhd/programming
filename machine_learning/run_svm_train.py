from __future__ import print_function
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import numpy as np

train_obj = serial.load_train_file('pylearn_svm.yaml')

train_obj.main_loop()

W, b = train_obj.model.get_param_values()
print(W)
print(b)

X = train_obj.dataset.X
x = np.linspace(-10, 10, 200)
f_x = (-1.0 * W[0][0] / W[1][0]) * x - (b[0][0] / W[1][0])
plt.plot(x, f_x)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
