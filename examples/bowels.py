import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from package.neunet import *

nn = NeuNet()
nn.load("mnist")


y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
y = y.reshape(10, 1)
nn.untrain(y, 0.05, 100)
