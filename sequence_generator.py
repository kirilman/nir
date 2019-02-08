import numpy as np
import random
from myutils import rename_state
def gaussian_generator(mean, deviation, sample_size):
    data = np.array([np.random.normal(mean,deviation) for x in range(sample_size)])
    return data

STAGE_DICT = {chr(x): x - 97 for x in range(97,123)}

class Sequence:
    def __init__(self, n, alphabet = [], period = 0, type = None, p = None, params = None, mean = None , varience = None):
        self.n = n
        self.path = None
        self.period = period
        self.alphabet = sorted(alphabet)
        self.mean = mean
        self.varience = varience
        # print(self.alphabet)
        self.type = type
        # self.seq = np.zeros((n,),dtype=np.int64)
        self.sequence = []
        if type == 'period':
            self.periodic()
        if type == 'random':
            self.random(p)
        if type == 'signal':
            self.sequence, stages = self.signal_()
        if type == 'test_discret':
            self.test_discrete(params)
        if type == 'continue':
            self.continue_signal(mean,varience, params)

    def continue_signal(self, mean, varince, params):
        _sequence = self.test_discret(params)
        self.path = [ STAGE_DICT[s] for s in _sequence]
        self.sequence = [ np.random.normal(mean[i],varince[i]) for i in self.path]

    def random(self,p):
        h = np.zeros((len(p) + 1, 2))
        for i in range(len(p)+1):
            if i == 0:
                h[i,0] = 0; h[i,1] = p[i]
            else:
                if i == len(p):
                    h[i,0] = p[i-1]; h[i,1] = 1
                else:
                    h[i,0] = p[i-1]; h[i,1] = p[i]
        # print(h)

        for _ in range(self.n):
            r = random.uniform(0, 1)
            for i in range(h.shape[0]):
                if r >= h[i,0] and r < h[i,1]:
                    self.sequence += self.alphabet[i]
                    continue


    def test_discrete(self,params=None):
        if params == None:
            params = {'a': {'len': [2, 5], 'depend_on': False},
                      'b': {'len': [2, 10], 'depend_on': False},
                      'c': {'len': [2, 7], 'depend_on': False},
                      'd': {'len': [1, 5], 'depend_on': False },
                      'e': {'len': [1, 3], 'depend_on': True}}
        length = 0

        # print(params)
        for key, item in params.items():
            length += max(item['len'])
            
        count_cycle = round(self.n/length)

        is_first = True

        rest = self.n
        while rest>0:
            for s in self.alphabet:
                if is_first == True:   #Обработка для первого элемента списка
                    r = np.random.choice(range(params[s]['len'][0], params[s]['len'][1] + 1))
                    rest-=r
                    for _ in range(r):
                        self.sequence.append(s)
                    is_first = False
                else:
                    if params[s]['depend_on'] == self.sequence[-1]:
                        continue
                    else:
                        r = np.random.choice( range(params[s]['len'][0], params[s]['len'][1] + 1))
                        if rest < r:
                            for _ in range(rest):
                                self.sequence.append(s)
                        else:
                            for _ in range(r):
                                self.sequence.append(s)
                        rest-=r
        
        # for i in enumerate(range(count_cycle)):
        #     for s in self.alphabet:
        #         if is_first == True:   #Обработка для первого элемента списка
        #             r = np.random.choice(range(params[s]['len'][0], params[s]['len'][1] + 1))
        #             for _ in range(r):
        #                 self.sequence.append(s)
        #             is_first = False
        #         else:
        #             if params[s]['depend_on'] == self.sequence[-1]:
        #                 continue
        #             else:
        #                 r = np.random.choice( range(params[s]['len'][0], params[s]['len'][1] + 1))
        #                 for _ in range(r):
        #                     self.sequence.append(s)
        self.path = [ STAGE_DICT[s] for s in self.sequence]
        self.n = len(self.sequence)

    def periodic(self):
        m = self.n // len(self.alphabet)
        rest = self.n % len(self.alphabet)
        for i in range(m):
            for k,s in enumerate(self.alphabet):
                self.sequence += self.alphabet[k]
            if i == m - 1:
                for k in range(rest):
                    self.sequence += self.alphabet[k]

    def anormal(self,p):
        x = self.sequence.copy()
        n = round(self.n*p)
        for i in range(n):
            index = round(np.random.uniform(0,self.n-1))
            x[index] = self.get_random_simbol()
        return x

    def to_int(self):
        seq = self.sequence.copy()
        x = []
        for s in seq:
            x+=[ord(s)%96]
        return x


    def get_random_simbol(self):
        s = round(np.random.uniform(len(self.alphabet))-1)
        return self.alphabet[s]

    def signal_(self):
        len_random_seq = 1
        data = np.array([])
        current_stage = 0
        seq_stages = []
        for _ in range(self.n):
            if current_stage == 0:
                a = np.random.uniform()
                if a >= 0 and a < 0.1:
                    temp = self.alphabet[0]
                    current_stage = 0
                elif a >= 0.1 and a < 0.5:
                    temp = self.alphabet[1]
                    current_stage = 1
                elif a >= 0.5 and a <= 1:
                    temp = self.alphabet[2]
                    current_stage = 2
            elif current_stage == 1:
                a = np.random.uniform()
                if a >= 0 and a <= 0.1:
                    temp = self.alphabet[0]
                    current_stage = 0
                elif a > 0.1 and a <= 0.2:
                    temp = self.alphabet[1]
                    current_stage = 1
                elif a > 0.2 and a <= 1:
                    temp = self.alphabet[2]
                    current_stage = 2
            else:
                a = np.random.uniform()
                if a >= 0 and a <= 0.1:
                    temp = self.alphabet[0]
                    current_stage = 0
                elif a > 0.1 and a <= 0.2:
                    temp = self.alphabet[1]
                    current_stage = 1
                elif a > 0.2 and a <= 1:
                    current_stage = 2
            if len(data) == 0:
                data = temp
            else:
                data = np.append(data, temp)
            seq_stages += [current_stage]
        return list(data), seq_stages

class Signal(Sequence):
    def __init__(self, n, count_stage, mean, varience, t ):
        super().__init__(n)
        self.n = n
        self.count_stage = count_stage
        self.mean = mean
        self.varience = varience
        self.t = t
        self.sequence, self.path = self.create_signal()

    def create_signal (self):
        l = 0
        sequence = []
        path = []
        current_stage = 0
        while(True):
            n = int(np.random.uniform(self.t[0], self.t[1]))
            l += n
            if l < self.n:
                path = path + [current_stage]*n
                sequence = sequence + np.random.normal(self.mean[current_stage], self.varience[current_stage], n).tolist()
                if current_stage < self.count_stage - 1:
                    current_stage+=1
                else:
                    current_stage=0
            else:
                n = self.n - len(path)
                path = path + [current_stage] * n
                sequence = sequence + np.random.normal(self.mean[current_stage], self.varience[current_stage],
                                                       n).tolist()
                break
        return sequence, path

class Continue_Signal():
    def __init__(self, _n, _mean, _varience):
        self.n = _n
        self.mean = _mean  
        self.varience = _varience


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

