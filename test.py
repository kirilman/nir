from pomegranate import *
import sequence_generator as generator
import matplotlib.pyplot as plt
import numpy as np
import myutils


test = generator.Sequence(10000, ['a','b'], type = 'continue',
                          params={'a': {'len': [1000, 1010], 'depend_on': False},
                                  'b': {'len': [25, 35], 'depend_on': False}},
                          mean = [0, 0.2] , variance = [0.03, 0.05])
fig = plt.figure(figsize = (20, 4))
plt.plot(test.sequence)
# plt.plot(test.path)
model = HiddenMarkovModel.from_samples(NormalDistribution,2,[test.sequence],
                                       labels = [list(map(lambda x: myutils.rename_state(x), test.path))], 
                                       algorithm = 'labeled')
myutils.print_model_distribution(model)