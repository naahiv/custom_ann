from model import *
from time import perf_counter

net = Network()
l1 = InputLayer(size=4)
l2 = HiddenLayer(size=3)
l3 = HiddenLayer(size=2)
l4 = OutputLayer(size=2)
net.add_layer(l1)
net.add_layer(l2)
net.add_layer(l3)
net.add_layer(l4)
net.initialize()

def_inp = np.array([1.5, -2.6, 0.85, 0.04])
p1 = perf_counter()
val = net.output(def_inp)
print(val, perf_counter()-p1)
