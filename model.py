from sequence_generator import  Sequence
import numpy as np
import matplotlib.pyplot as plt
from pomegranate import *
import warnings
import time

s = Sequence(round(np.random.uniform(20,67)),['a','c','b','d','e','f','g'],type='period')
s = Sequence(250,['a','b','c','f'],type='period')

sequence = np.array(s.sequence).reshape(-1,1)
print(sequence.shape)
fig = plt.figure()
# plt.plot(sequence)
#print(sequence)


model = HiddenMarkovModel()
print(sequence.shape)
c = np.concatenate((sequence,sequence))
#print('c={}'.format(c))

model = model.from_samples(distribution = DiscreteDistribution, n_components = 4,X = sequence,max_iterations=200,
                          labels = sequence)
model = model.fit(sequence)
# model.bake()
# model.bake()

fig1 = plt.figure()
# plt.ishold()
# model.plot()
plt.plot(s.sequence)
plt.show()
test = np.array(s.sequence)
print(test.shape)
print(model.log_probability(test))

#unormal
print(model.log_probability(s.unormal(0.5)))

print('Len p=',len(model.predict(test)))
plt.plot(s.unormal(0.5))
plt.show()


print('Проверка предсказания')
print(model.log_probability(test))
print(model.log_probability(s.unormal(0.5)))



an = ['a' for i in range(len(sequence))]
# an = np.array(an).reshape(1,-1)
# print(an)
# print(an.shape)
print(test.shape)
with open('model.txt','w') as file:
    file.write(str(model.__getstate__()))
    file.write('\n')
    file.write('Проверка предсказания')
    file.write(str(model.log_probability(test)))
    file.write(str(model.log_probability(an)))
    file.write(str(model.sample(125))+'\n')
    file.write(str(model.dense_transition_matrix()))
fig = plt.figure('test')
plt.plot(model.sample(100))
plt.show()

# import pomegranate as pom
# import numpy as np
#
# polA_pri = list('VISYDNYVT')
# polA_sec = list('NNNBBBBBB')
# secondary_str_types = ['N','B']
# X=np.array(polA_pri)[:,None]
# print(X)
# print(X.shape)
# model = pom.HiddenMarkovModel.from_samples(pom.DiscreteDistribution, n_components=len(secondary_str_types), X=np.array(polA_pri)[:,None], labels=np.array(polA_sec)[:,None], algorithm='labeled')