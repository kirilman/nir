from pomegranate import *
import sequence_generator as sequence

s = sequence.Sequence(100,['a','b','c'],type = 'signal')
print(s.sequence)