import numpy as np
import random

def gaussian_generator(mean, deviation, sample_size):
    data = np.array([np.random.normal(mean,deviation) for x in range(sample_size)])
    return data

class Sequence:
    def __init__(self, n, alphabet, period = 0, type = None, p = None):
        self.n = n
        self.period = period
        self.alphabet = sorted(alphabet)
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

    def random(self,p):
        h = np.zeros((len(p) + 1, 2))
        for i in range(len(p)+1):
            print(i)
            if i == 0:
                h[i,0] = 0; h[i,1] = p[i]
            else:
                if i == len(p):
                    h[i,0] = p[i-1]; h[i,1] = 1
                else:
                    h[i,0] = p[i-1]; h[i,1] = p[i]
        print(h)

        for _ in range(self.n):
            r = random.uniform(0, 1)
            for i in range(h.shape[0]):
                if r >= h[i,0] and r < h[i,1]:
                    self.sequence += self.alphabet[i]
                    continue


    def periodic(self):
        m = self.n // len(self.alphabet)
        rest = self.n % len(self.alphabet)
        for i in range(m):
            for k,s in enumerate(self.alphabet):
                self.sequence += self.alphabet[k]
            if i == m - 1:
                for k in range(rest):
                    self.sequence += self.alphabet[k]

    def unormal(self,p):
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