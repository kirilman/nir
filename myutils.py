import numpy as np
import json
def frequency_occurrence(a,show=True):
    unique = set(a)
    n = len(a)
    d = {}
    for u in unique:
        d.update({u:a.count(u)/n})
    if show == True:
        print(d)
    return d


def table_from_MarkovChain(markov_model):
    obj = markov_model.distributions[1].to_json()
    obj = json.loads(obj)
    return  obj['table']


def get_path_with_viterbi(model):
    path = model.viterbi()
    return path
# if __name__ == "__main__":
#     a = ['a','b','c','d','b']
#
#     frequency_occurrence(a,False)


