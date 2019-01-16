import sequence_generator as generator
import numpy as np
import matplotlib.pyplot as plt
from pomegranate import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='matplotlib')
import gc

for i in range(5):
    alpha = ['a','b','c','d','e']
    sequence = generator.Sequence(10,alpha,type='test_discret',p=[0.05,0.1,0.4,0.8])
    s = sequence.sequence.copy()
    model = HiddenMarkovModel(name = 'test').from_samples(DiscreteDistribution,3,X = [s])
    print(id(model))
    print(model.name)
    print(model.model)

    del model
    gc.collect()
    print(s)
