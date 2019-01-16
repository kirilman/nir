from pomegranate import *
import sequence_generator as generator
import matplotlib.pyplot as plt
import numpy as np

# def rename_state(x):
#     a = 's'+x
#     return a
# # for i in range(3):
# #     s = sequence.Signal(1000,3,[5,2,3],[0.5,3,1],[50,100])
# #     s = s.sequence
# #     plt.plot(s)
# #         # plt.xscale(['a','b','c','d'])
# # plt.show()
# # print(s)
# s = generator.Signal(500,4,[10,10,10,12],[0.5,2,4,2],[50,100])
# seq = s.sequence
# print(seq[:5])
# v = int(len(seq)*0.8)
# x_train, x_test = seq[0:v], seq[v:]
# path_train = s.path[0:v]
# path_source = s.path[v:]

# labels = list(map(rename_state,list(map(str,path_train))))
# print(len(labels))
# model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = 4,X = [x_train],
#                                        labels=[labels],algorithm='labeled' )
# p = model.viterbi(x_test)
# path_test = []
# for i in p[1]:
#     path_test +=[i[0]]
# path_test = path_test[1:]


# print('Длина пути для теста: soure={}, test={}'.format(len(path_source), len(path_test)))

# path_source = np.array(path_source)
# path_test = np.array(path_test)
# path_train = np.array(path_train)
# fig,ax = plt.subplots(3,1,dpi=140)
# ax[0].plot(x_train)
# ax[0].plot(path_train*3,'g',lw=0.9)

# p = model.viterbi(x_train)
# path_s = []

# for i in p[1]:
#     path_s +=[i[0]]
# path_s = path_s[1:]
# path_s = np.array(path_s)
# ax[0].plot(path_s*3,'r')

# ax[2].hist(x_train,bins = int(np.log(len(x_train))+30));

# ax[1].plot(x_test)
# ax[1].plot(path_source*3,'g')
# ax[1].plot(path_test*3,'r--')
# plt.tight_layout()
# print(np.mean(seq))
# print(model)
# plt.show()
# signal = np.array(seq)'


def get_slice(s):
    m = np.random.choice(range(len(s)))
    # m = 0
    print(m)
    c = s[m]
    start, stop = 0,0
    i = 1 
    flag_1 = True
    flag_2 = True
    while((flag_1==True) or (flag_2 == True)):
        if flag_1 == True:
            if m-i-1 == 0:
                start = m-i+1
                flag_1 = False
            else:
                if s[m - i]!=c:
                    start = m-i+1
                    flag_1 = False
        if flag_2 == True:
            if m + i - 1 == len(s) - 1:
                stop = m + i
                flag_2 = False
            else:
                if s[m+i]!=c:
                    stop = m + i
                    flag_2 = False
        i+=1
        print('i ',i)
    return start, stop   

# s = ['e','a','a','a','b','c','c','c','c','c','c','d','d','d','d','d','d']

alpha = ['a','b','c','d','e']
N = 30
sequence = generator.Sequence(N,alpha,type='test_discret',p=[0.05,0.1,0.4,0.8])
s = sequence.sequence.copy()
print(s)
for i,_ in enumerate(s):
    if i == 0:
        print('[',end = ' ')
    print(i,end='   ,')
print()
start, stop = get_slice(s)
print(start, stop)

print(s[start:stop])


a = 510
del a
print(a)