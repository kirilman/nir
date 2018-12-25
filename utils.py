import numpy as np


def frequency_occurrence(a,show=True):
    unique = set(a)
    n = len(a)
    d = {}
    for u in unique:
        d.update({u:a.count(u)/n})
    if show == True:
        print(d)
    return d

# if __name__ == "__main__":
#     a = ['a','b','c','d','b']
#
#     frequency_occurrence(a,False)
