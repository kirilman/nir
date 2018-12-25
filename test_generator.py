from sequence_generator import  Sequence
import numpy as np
import matplotlib.pyplot as plt
from pomegranate import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='matplotlib')
#
# def periodic(len, simbols):
#     step  = 5
#     seq = np.zeros([len,0])
#     k = 0
#     while (k>len):
#         for s in simbols:
#             seq[k] = [s; k++ for i in range(step)]
#     return seq
#
# periodic(10,['a','b'])
# for i in range(100):

# for i in range(10):
#     s = Sequence(round(np.random.uniform(20,67)),['a','c','b','d','e','f','g'],type='period')
#     plt.plot(s.seq)
#     plt.show()

s = Sequence(round(np.random.uniform(20,67)),['a','c','b','d','e','f','g'],type='period')
plt.plot(s.sequence)
print(s.sequence)
fig = plt.figure()
print(len(s.sequence))
# plt.plot(s.seq)
# plt.show()


# from pomegranate import *
s1 = State(NormalDistribution(5, 1),name="state1")
s2 = State(NormalDistribution(1, 7),name="state2")
s3 = State(NormalDistribution(8, 2),name="state3")
model = HiddenMarkovModel()
model.add_states(s1, s2, s3)
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.7)
model.add_transition(s1, s2, 0.3)
model.add_transition(s2, s1, 0.5)
model.add_transition(s2, s2, 0.8)
model.add_transition(s2, s3, 0.2)
model.add_transition(s3, s3, 0.9)
model.add_transition(s3, model.end, 0.1)
model.bake()


# plt.ishold()
model.plot()
plt.show()



